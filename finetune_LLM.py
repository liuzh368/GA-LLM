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

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    poi_embedding_path: str = field(default=None, metadata={"help": "Path to the POI embeddings (.npy file)."})
    projector_path: str = field(default=None, metadata={"help": "Path to the pre-trained projector (linear layer)."})

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
        model: transformers.PreTrainedModel
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

def initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, linear_mapping_layer):
    """Initialize the POI embedding layer with pre-trained POI embeddings and apply the linear mapping."""
    with torch.no_grad():
        device = poi_embedding_layer.weight.device  # 获取 poi_embedding_layer 所在的设备
        # 将 POI embeddings 移动到同一个设备上
        poi_embeddings = poi_embeddings.to(device)
        # 将128维的POI embedding映射到4096维
        poi_embeddings = linear_mapping_layer(poi_embeddings).to(device)  # 将线性映射后的 embeddings 也移动到同一设备
        poi_embedding_layer.weight.data = poi_embeddings
    print("POI embeddings successfully initialized.")

def embedding_hook(module, inputs, output):
    """钩子函数，用于处理 POI token 的 embedding 处理"""
    input_ids = inputs[0].to(module.weight.device)  # 确保 input_ids 和 embedding layer 在相同的设备上
    poi_embedding_layer = module.poi_embedding_layer.weight
    poi_token_start_id = module.poi_token_start_id

    # print(f"Embedding Hook Called. Input IDs: {input_ids}")

    is_poi_token = input_ids >= poi_token_start_id
    token_embedding = output.clone()

    poi_token_ids = input_ids[is_poi_token] - poi_token_start_id
    if poi_token_ids.size(0) > 0:
        # print(f"Embedding Hook Called. Input IDs: {input_ids}")
        poi_token_ids = poi_token_ids.to(poi_embedding_layer.device)  # 移动到相同设备
        poi_token_embedding = poi_embedding_layer[poi_token_ids]
        token_embedding[is_poi_token] = poi_token_embedding.to(token_embedding.dtype)
        # print(f"Replaced POI token embeddings at positions: {torch.nonzero(is_poi_token).squeeze()}")

    return token_embedding

def apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id):
    """给模型的 embedding 层添加钩子函数"""
    embedding_layer = model.get_input_embeddings()
    embedding_layer.poi_embedding_layer = poi_embedding_layer
    embedding_layer.poi_token_start_id = poi_token_start_id
    embedding_layer.register_forward_hook(embedding_hook)

def load_projector(projector_path, input_dim=128, output_dim=4096):
    """Load the pre-trained projector (linear mapping layer)."""
    linear_mapping_layer = nn.Linear(input_dim, output_dim)
    state_dict = torch.load(projector_path)
    linear_mapping_layer.load_state_dict(state_dict)
    return linear_mapping_layer

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

    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # 加载 POI embeddings 和 Projector
    poi_embeddings = load_poi_embeddings(data_args.poi_embedding_path)
    linear_mapping_layer = load_projector(data_args.projector_path)

    # 检查当前的最大 Token ID
    max_token_id = max(tokenizer.get_vocab().values())  # 获取词汇表中最大的 Token ID
    poi_token_start_id = max_token_id + 1
    num_poi_tokens = len(poi_embeddings)

    # 创建专用的POI embedding层，其维度应与大模型的 token embedding 一致 (4096 维)
    model_embedding_dim = model.get_input_embeddings().weight.shape[1]
    poi_embedding_layer = nn.Embedding(num_embeddings=num_poi_tokens, embedding_dim=model_embedding_dim)

    # 初始化POI token的embedding，并使用线性映射层将其转换为 4096 维
    initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, linear_mapping_layer)

    # 添加 POI tokens 到 tokenizer 的词汇表中
    # poi_tokens = [f"<POI {i}>" for i in range(0, num_poi_tokens)] # 对于nyc数据集的处理与ca和tky不同，因为起始POI id是从0开始
    # 根据数据集名称决定 POI tokens 的生成方式
    if "nyc" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(0, num_poi_tokens)]
    elif "ca" in data_args.data_path.lower() or "tky" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(1, num_poi_tokens + 1)]
    else:
        raise ValueError("Unsupported dataset name. Please use 'nyc', 'ca', or 'tky'.")

    special_tokens_dict = {"additional_special_tokens": poi_tokens}

    # 调整tokenizer和embedding大小
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    # 应用钩子函数，处理 POI token embedding
    apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id)

    # 冻结模型中的所有权重，确保只有 LoRA 和线性映射层更新
    for param in model.parameters():
        param.requires_grad = False

    for param in linear_mapping_layer.parameters():
        param.requires_grad = False  # 冻结 projector

    # 启用LoRA进行低秩微调
    if training_args.low_rank_training:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 对 q_proj, k_proj, v_proj 和 o_proj 进行微调
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # 启用额外的可训练参数
        [p.requires_grad_() for n, p in model.named_parameters() if
         any([k in n for k in training_args.trainable_params.split(",")])]

    # 强制将模型输出转为 float32
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    # Trainer进行微调
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # 禁用缓存，启用梯度检查点
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_module['data_collator'],
        train_dataset=data_module['train_dataset'],
    )

    # 开始训练
    trainer.train(resume_from_checkpoint=False)

    torch.cuda.empty_cache()

    # 保存模型和状态
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()