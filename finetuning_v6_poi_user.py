import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import numpy as np
from llama_attn_replace_sft import replace_llama_attn  # 添加这一行，导入 replace_llama_attn

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
os.environ["WANDB_DISABLED"] = "true"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2":(
        "<s>[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_input_llama2": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} \n{input} [/INST]"
    )
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    poi_embedding_path: str = field(default=None, metadata={"help": "Path to the POI embeddings (.npy file)."})
    user_embedding_path: str = field(default=None, metadata={"help": "Path to the User embeddings (.npy file)."})  # 添加user embedding路径

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether to use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of update steps to accumulate before performing a backward/update pass."},
    )
    deepspeed: str = field(
        default="ds_configs/stage2.json",
        metadata={"help": "Path to the DeepSpeed config file."}
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX  # 标记 `question` 部分不计算损失
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")
        sources = ['<question>:' + example["question"] for example in list_data_dict]
        targets = ['<answer>:' + f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def load_poi_embeddings(poi_embedding_path):
    """Load the POI embeddings from a .npy file."""
    poi_embeddings = np.load(poi_embedding_path)
    return torch.tensor(poi_embeddings)

def load_user_embeddings(user_embedding_path):
    """Load the User embeddings from a .npy file."""  # 新增函数，加载user embeddings
    user_embeddings = np.load(user_embedding_path)
    return torch.tensor(user_embeddings)

def initialize_poi_embeddings(model, tokenizer, poi_embeddings):
    """Directly initialize the POI tokens embeddings without any mapping."""
    embedding_layer = model.get_input_embeddings()
    embedding_weight = embedding_layer.weight.data

    # 将 POI embedding 转移到模型所在的设备
    poi_embeddings = poi_embeddings.to(model.device)  # 确保 poi_embeddings 在相同的设备上

    # 将 POI token 转化为对应的 token id
    poi_token_ids = [tokenizer.convert_tokens_to_ids(f"<POI {i}>") for i in range(1, len(poi_embeddings) + 1)]

    # 确保 embedding 层的尺寸已经更新
    assert embedding_weight.size(0) >= len(poi_token_ids), "Embedding layer size mismatch"

    # 替换 POI token embedding
    for idx, poi_id in enumerate(poi_token_ids):
        embedding_weight[poi_id] = poi_embeddings[idx]

    print("POI embeddings successfully initialized in the LLM's embedding layer.")

    # 释放 POI embeddings 的内存
    del poi_embeddings
    torch.cuda.empty_cache()  # 手动清理缓存的显存

def initialize_user_embeddings(model, tokenizer, user_embeddings):
    """Directly initialize the User tokens embeddings without any mapping."""  # 新增函数，初始化user embeddings
    embedding_layer = model.get_input_embeddings()
    embedding_weight = embedding_layer.weight.data

    # 将 user embedding 转移到模型所在的设备
    user_embeddings = user_embeddings.to(model.device)

    # 将 user token 转化为对应的 token id
    user_token_ids = [tokenizer.convert_tokens_to_ids(f"<user {i}>") for i in range(1, len(user_embeddings) + 1)]

    # 确保 embedding 层的尺寸已经更新
    assert embedding_weight.size(0) >= len(user_token_ids), "Embedding layer size mismatch"

    # 替换 user token embedding
    for idx, user_id in enumerate(user_token_ids):
        embedding_weight[user_id] = user_embeddings[idx]

    print("User embeddings successfully initialized in the LLM's embedding layer.")

    # 释放 user embeddings 的内存
    del user_embeddings
    torch.cuda.empty_cache()  # 手动清理缓存的显存


def train():
    # 解析命令行参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 替换 LLAMA 模型的注意力机制
    replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # 加载模型的配置文件
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    # 设置RoPE缩放因子
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # 加载模型，使用4位量化（QLoRA）
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    # 加载POI和User embeddings
    poi_embeddings = load_poi_embeddings(data_args.poi_embedding_path)
    user_embeddings = load_user_embeddings(data_args.user_embedding_path)  # 加载user embedding

    poi_tokens = [f"<POI {i}>" for i in range(1, len(poi_embeddings) + 1)]
    user_tokens = [f"<user {i}>" for i in range(1, len(user_embeddings) + 1)]  # 定义user token
    special_tokens_dict = {"additional_special_tokens": poi_tokens + user_tokens}

    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # 检查并调整 tokenizer 的特殊标记（如填充、起始、结束和未知标记）
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # 调整tokenizer和embedding大小
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    # 初始化POI和User token的embedding，并释放内存
    with torch.no_grad():
        initialize_poi_embeddings(model, tokenizer, poi_embeddings)
        initialize_user_embeddings(model, tokenizer, user_embeddings)  # 初始化user embedding

    # 冻结模型中的所有权重，确保只有适配器（LoRA层）更新
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    # 创建数据模块
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # 启用低秩训练（LoRA）
    if training_args.low_rank_training:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

        # 启用额外的可训练参数
        [p.requires_grad_() for n, p in model.named_parameters() if
         any([k in n for k in training_args.trainable_params.split(",")])]

    # 强制将模型输出转为 float32
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    # Verifying the datatypes
    # 统计模型中各层权重的数据类型，并输出每种数据类型所占的比例
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()

    total = sum(dtypes.values())
    for dtype, count in dtypes.items():
        print(f"Data type: {dtype}, Count: {count}, Percentage: {count / total * 100:.2f}%")

    # 禁用缓存，启用梯度检查点
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Trainer进行微调
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    print("Training starts...")
    print(f"POI embeddings shape: {poi_embeddings.shape}")
    print(f"User embeddings shape: {user_embeddings.shape}")  # 输出user embeddings形状
    print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
    print(f"First batch input ids: {next(iter(data_module['train_dataset']))['input_ids']}")

    # 开始训练
    trainer.train(resume_from_checkpoint=False)

    # 保存模型和状态
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()