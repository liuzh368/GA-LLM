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
from llama_attn_replace_sft import replace_llama_attn  # 导入 replace_llama_attn

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
    gps_embedding_path: str = field(default=None, metadata={"help": "Path to the GPS embeddings (.npy file)."})
    poi_mapping_layer_path: str = field(default=None, metadata={"help": "Path to the pre-trained poi_mapping_layer."})
    gps_mapping_layer_path: str = field(default=None, metadata={"help": "Path to the pre-trained gps_mapping_layer."})

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

def load_embeddings(embedding_path):
    """Load embeddings from a .npy file."""
    embeddings = np.load(embedding_path)
    return torch.tensor(embeddings)

def load_mapping_layer(mapping_layer_path, input_dim=128, output_dim=4096):
    """Load the pre-trained mapping layer (e.g., poi_mapping_layer or gps_mapping_layer)."""
    # 定义与训练时相同的三层结构
    mapping_layer = nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.Linear(output_dim, output_dim)
    )
    # 加载合并后的权重
    state_dict = torch.load(mapping_layer_path)
    # 处理 state_dict 的键名
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'layer_' in key:
            layer_num = key.split('_')[1]
            param_type = key.split('_')[2]
            if param_type == 'type':
                continue  # 跳过 'layer_{i}_type'
            new_key = f"{int(layer_num)}.{param_type}"
            new_state_dict[new_key] = value
    mapping_layer.load_state_dict(new_state_dict)
    return mapping_layer

def initialize_embeddings(embedding_layer, embeddings, device):
    """Initialize the embedding layer with raw embeddings."""
    with torch.no_grad():
        embedding_layer.to(device)
        embeddings = embeddings.to(device)
        # 直接将原始的 embeddings 存入 embedding_layer
        embedding_layer.weight.data = embeddings
    print("Initialized embeddings.")

def embedding_hook(module, inputs, output):
    """Embedding hook function to process POI and GPS token embeddings."""
    input_ids = inputs[0].to(module.weight.device)
    poi_embedding_layer = module.poi_embedding_layer.weight
    gps_embedding_layer = module.gps_embedding_layer.weight
    poi_token_start_id = module.poi_token_start_id
    gps_token_start_id = module.gps_token_start_id
    poi_mapping_layer = module.poi_mapping_layer
    gps_mapping_layer = module.gps_mapping_layer

    is_poi_token = (input_ids >= poi_token_start_id) & (input_ids < gps_token_start_id)
    is_gps_token = input_ids >= gps_token_start_id
    token_embedding = output.clone()

    # Process POI tokens
    poi_token_ids = input_ids[is_poi_token] - poi_token_start_id
    if poi_token_ids.size(0) > 0:
        poi_token_ids = poi_token_ids.to(poi_embedding_layer.device)
        # Get raw embeddings
        original_poi_embeddings = poi_embedding_layer[poi_token_ids]
        # Map them to higher dimensions using poi_mapping_layer
        poi_token_embedding = poi_mapping_layer(original_poi_embeddings)
        token_embedding[is_poi_token] = poi_token_embedding.to(token_embedding.dtype)

    # Process GPS tokens
    gps_token_ids = input_ids[is_gps_token] - gps_token_start_id
    if gps_token_ids.size(0) > 0:
        gps_token_ids = gps_token_ids.to(gps_embedding_layer.device)
        # Get raw embeddings
        original_gps_embeddings = gps_embedding_layer[gps_token_ids]
        # Map them to higher dimensions using gps_mapping_layer
        gps_token_embedding = gps_mapping_layer(original_gps_embeddings)
        token_embedding[is_gps_token] = gps_token_embedding.to(token_embedding.dtype)

    return token_embedding

def apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id, poi_mapping_layer,
                         gps_embedding_layer, gps_token_start_id, gps_mapping_layer):
    """Add the embedding hook to the model's embedding layer."""
    embedding_layer = model.get_input_embeddings()
    embedding_layer.poi_embedding_layer = poi_embedding_layer.to(embedding_layer.weight.device)
    embedding_layer.poi_token_start_id = poi_token_start_id
    embedding_layer.poi_mapping_layer = poi_mapping_layer.to(embedding_layer.weight.device)
    embedding_layer.gps_embedding_layer = gps_embedding_layer.to(embedding_layer.weight.device)
    embedding_layer.gps_token_start_id = gps_token_start_id
    embedding_layer.gps_mapping_layer = gps_mapping_layer.to(embedding_layer.weight.device)
    embedding_layer.register_forward_hook(embedding_hook)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 POI 和 GPS embeddings
    poi_embeddings = load_embeddings(data_args.poi_embedding_path)
    gps_embeddings = load_embeddings(data_args.gps_embedding_path)

    num_poi_tokens = len(poi_embeddings)
    num_gps_tokens = len(gps_embeddings)

    # 创建专用的 POI 和 GPS embedding 层，其维度应为 128
    poi_embedding_layer = nn.Embedding(num_embeddings=num_poi_tokens, embedding_dim=128)
    gps_embedding_layer = nn.Embedding(num_embeddings=num_gps_tokens, embedding_dim=128)

    # 初始化 POI 和 GPS embeddings
    initialize_embeddings(poi_embedding_layer, poi_embeddings, device)
    initialize_embeddings(gps_embedding_layer, gps_embeddings, device)

    # 加载预训练的 poi_mapping_layer 和 gps_mapping_layer
    poi_mapping_layer = load_mapping_layer(data_args.poi_mapping_layer_path)
    gps_mapping_layer = load_mapping_layer(data_args.gps_mapping_layer_path)
    poi_mapping_layer.to(device)
    gps_mapping_layer.to(device)

    # 检查当前的最大 Token ID
    max_token_id = max(tokenizer.get_vocab().values())
    poi_token_start_id = max_token_id + 1
    gps_token_start_id = poi_token_start_id + num_poi_tokens

    # 添加 POI tokens 到 tokenizer 的词汇表中，根据数据集进行处理
    if "nyc" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(0, num_poi_tokens)]
        gps_tokens = [f"<GPS {i}>" for i in range(0, num_gps_tokens)]
    elif "ca" in data_args.data_path.lower() or "tky" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(1, num_poi_tokens + 1)]
        gps_tokens = [f"<GPS {i}>" for i in range(1, num_gps_tokens + 1)]
    else:
        raise ValueError("Unsupported dataset name. Please use 'nyc', 'ca', or 'tky'.")

    special_tokens_dict = {"additional_special_tokens": poi_tokens + gps_tokens}

    # 调整tokenizer和embedding大小
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    # 应用钩子函数，处理 POI 和 GPS token embedding
    apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id, poi_mapping_layer,
                         gps_embedding_layer, gps_token_start_id, gps_mapping_layer)

    # 冻结模型中的所有权重
    for param in model.parameters():
        param.requires_grad = False

    # 冻结 embedding layers 和 mapping layers
    for param in poi_embedding_layer.parameters():
        param.requires_grad = False
    for param in gps_embedding_layer.parameters():
        param.requires_grad = False
    for param in poi_mapping_layer.parameters():
        param.requires_grad = False
    for param in gps_mapping_layer.parameters():
        param.requires_grad = False

    # 启用LoRA进行低秩微调
    if training_args.low_rank_training:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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