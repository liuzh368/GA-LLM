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
from llama_attn_replace_sft import replace_llama_attn  # 替换LLAMA注意力机制

from transformers import TrainerCallback
import random
# 设置随机种子，确保可复现性
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法相同
    torch.backends.cudnn.benchmark = False

# 在代码的开头调用 set_seed 函数
set_seed(42)  # 42 是常用的默认随机种子，可以根据需要修改

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
        metadata={"help": "Whether to use low rank adaptation (LoRA) for training."},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of update steps to accumulate before performing a backward/update pass."},
    )
    deepspeed: str = field(
        default="ds_configs/stage2.json",
        metadata={"help": "Path to the DeepSpeed config file."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "Rank for the LoRA low-rank matrix approximation."}
    )
    lora_alpha: float = field(
        default=32,
        metadata={"help": "Alpha scaling factor for combining LoRA weights."}
    )
    save_total_limit: int = field(
        default=3,  # 添加此参数来限制保留的 checkpoint 数量
        metadata={"help": "Total number of checkpoints to keep. Older checkpoints will be deleted."}
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


def initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, device):
    """初始化 POI embedding 层，但不直接映射到高维"""
    with torch.no_grad():
        poi_embedding_layer.to(device)
        poi_embeddings = poi_embeddings.to(device)

        # 将原始 128 维度的 POI embeddings 存入 poi_embedding_layer，但不映射
        poi_embedding_layer.weight.data = poi_embeddings
    print("Initialized 128-dimensional POI embeddings.")


def embedding_hook(module, inputs, output):
    """钩子函数，动态使用 linear_mapping_layer 将 POI embedding 映射到高维"""
    input_ids = inputs[0].to(module.weight.device)
    poi_token_start_id = module.poi_token_start_id

    is_poi_token = input_ids >= poi_token_start_id
    token_embedding = output.clone()

    poi_token_ids = input_ids[is_poi_token] - poi_token_start_id
    if poi_token_ids.size(0) > 0:
        # 从原始 128 维度 POI 嵌入开始，经过 linear_mapping_layer 动态映射到 4096 维
        original_poi_embedding = module.poi_embedding_layer(poi_token_ids)
        poi_token_embedding = module.linear_mapping_layer(original_poi_embedding)
        token_embedding[is_poi_token] = poi_token_embedding.to(token_embedding.dtype)
        # 输出 linear_mapping_layer 的部分权重，用于确认更新情况
        print("Sample of linear_mapping_layer weights:", module.linear_mapping_layer[0].weight.data[0][:5])

    return token_embedding


# 确保 embedding_hook 应用正确
def apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id):
    """给模型的 embedding 层添加钩子函数"""
    embedding_layer = model.get_input_embeddings()
    embedding_layer.poi_embedding_layer = poi_embedding_layer
    embedding_layer.poi_token_start_id = poi_token_start_id
    embedding_layer.linear_mapping_layer = model.linear_mapping_layer  # 添加映射层引用
    embedding_layer.register_forward_hook(embedding_hook)


# 输出指定 token ID 的对应文本
def check_token_ids(tokenizer, ids_to_check):
    print("Checking token ID mappings:")
    for token_id in ids_to_check:
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"Token ID {token_id}: {token}")


# 合并并保存包含 LoRA 权重的 linear_mapping_layer
def save_combined_linear_mapping_layer(model, output_path, alpha, r):
    """保存 linear_mapping_layer 的完整结构，包括激活层，合并 LoRA 权重后存为一个文件。"""
    with torch.no_grad():
        # 尝试直接访问 linear_mapping_layer
        if hasattr(model, 'linear_mapping_layer'):
            module = model.linear_mapping_layer
            print("Found linear_mapping_layer in model")
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'linear_mapping_layer'):
            module = model.base_model.linear_mapping_layer
            print("Found linear_mapping_layer in model.base_model")
        elif hasattr(model, 'base_model') and hasattr(model.base_model.model, 'linear_mapping_layer'):
            module = model.base_model.model.linear_mapping_layer
            print("Found linear_mapping_layer in model.base_model.model")
        else:
            raise ValueError("linear_mapping_layer not found in the model.")

        combined_weight_path = os.path.join(output_path, "linear_mapping_layer_combined.pt")

        # 创建一个字典存储层结构
        combined_state = {}

        for i, layer in enumerate(module):
            if isinstance(layer, nn.Linear):
                # 合并 LoRA 权重
                if hasattr(layer, "lora_A") and hasattr(layer, "lora_B"):
                    lora_weight = (alpha / r) * (
                        layer.lora_B["default"].weight @ layer.lora_A["default"].weight)
                    layer.weight.data += lora_weight

                # 保存 Linear 层权重和偏置
                combined_state[f"layer_{i}_weight"] = layer.weight.data.clone()
                if layer.bias is not None:
                    combined_state[f"layer_{i}_bias"] = layer.bias.data.clone()

            elif isinstance(layer, nn.GELU):
                # 保存激活层的类型
                combined_state[f"layer_{i}_type"] = "GELU"

        # 将完整的结构保存到一个文件
        torch.save(combined_state, combined_weight_path)
        print(f"Full linear_mapping_layer structure saved to: {combined_weight_path}")


class POITrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        """Override the checkpoint saving method to include linear_mapping_layer and manage checkpoints."""
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the model's state dict
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        print(f"Model state_dict saved to: {output_dir}/pytorch_model.bin")

        # Save the combined linear_mapping_layer
        save_combined_linear_mapping_layer(
            model=self.model,
            output_path=output_dir,
            alpha=self.args.lora_alpha,
            r=self.args.lora_r
        )

        # Manually save the optimizer and scheduler states
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        # Save the trainer state
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Rotate checkpoints, ensuring only the most recent ones are kept
        self._rotate_checkpoints(use_mtime=True, output_dir=self.args.output_dir)

def train():
    # 解析命令行参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 输出所有参数配置
    print("Model Arguments:")
    print(f"  Model Name or Path: {model_args.model_name_or_path}")
    print(f"  Model Type: {model_args.model_type}")

    print("\nData Arguments:")
    print(f"  Data Path: {data_args.data_path}")
    print(f"  POI Embedding Path: {data_args.poi_embedding_path}")

    print("\nTraining Arguments:")
    print(f"  Cache Directory: {training_args.cache_dir}")
    print(f"  Optimizer: {training_args.optim}")
    print(f"  Model Max Length: {training_args.model_max_length}")
    print(f"  Use Flash Attention: {training_args.use_flash_attn}")
    print(f"  Use Full Attention: {training_args.use_full_attn}")
    print(f"  Low Rank Training: {training_args.low_rank_training}")
    print(f"  Gradient Accumulation Steps: {training_args.gradient_accumulation_steps}")
    print(f"  DeepSpeed Config: {training_args.deepspeed}")
    print(f"  LoRA r: {training_args.lora_r}")
    print(f"  LoRA Alpha: {training_args.lora_alpha}")

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

    # 加载POI embeddings
    poi_embeddings = load_poi_embeddings(data_args.poi_embedding_path)

    num_poi_tokens = len(poi_embeddings)

    # 创建专用的POI embedding层，其维度应与大模型的 token embedding 一致 (4096 维)
    model_embedding_dim = model.get_input_embeddings().weight.shape[1]
    poi_embedding_layer = nn.Embedding(num_embeddings=num_poi_tokens, embedding_dim=model_embedding_dim)

    device = model.device  # 使用模型的设备

    linear_mapping_layer = nn.Sequential(
        nn.Linear(128, model_embedding_dim),
        nn.GELU(),  # 非线性激活函数，您可以替换为其他激活函数
        nn.Linear(model_embedding_dim, model_embedding_dim)
    ).to(device)

    # 将 linear_mapping_layer 添加到模型中作为一个属性
    # model.linear_mapping_layer = linear_mapping_layer
    # 将 linear_mapping_layer 添加到模型中作为一个子模块
    model.add_module('linear_mapping_layer', linear_mapping_layer)
    print("linear_mapping_layer registered:", "linear_mapping_layer" in dict(model.named_modules()).keys())
    print("Initial linear_mapping_layer weights:", model.linear_mapping_layer[0].weight.data[0][:5])

    # 初始化POI token的embedding，并使用线性映射层将其转换为 4096 维
    initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, device)
    # 检查当前的最大 Token ID
    max_token_id = max(tokenizer.get_vocab().values())  # 获取词汇表中最大的 Token ID
    poi_token_start_id = max_token_id + 1  # 设定 POI token 的起始 ID
    # 添加 POI tokens 到 tokenizer 的词汇表中
    # poi_tokens = [f"<POI {i}>" for i in range(0, num_poi_tokens)]
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

    # 检查 token ID 32000, 32001, 32002 对应的文本
    check_token_ids(tokenizer, [32000, 32001, 32002])

    # 应用钩子函数，处理 POI token embedding
    apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id)

    # 冻结模型中的所有权重，确保只有 projector 和线性映射层更新
    for param in model.parameters():
        param.requires_grad = False

    # 仅设置 model.linear_mapping_layer 的参数为可训练
    for param in model.linear_mapping_layer.parameters():
        param.requires_grad = True

    # 启用 LoRA 对 linear_mapping_layer 微调
    if training_args.low_rank_training:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["linear_mapping_layer.0", "linear_mapping_layer.2"],  # 指定单个 Linear 层
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # 检查 `linear_mapping_layer` 是否被 `LoRA` 识别
        for name, module in model.named_modules():
            if "linear_mapping_layer" in name:
                print(f"{name}: {module}")

        # 确保 lora_A 和 lora_B 的 requires_grad 为 True
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
                print(f"{name}.requires_grad: {param.requires_grad}")

    # 创建数据模块
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # 禁用缓存，启用梯度检查点
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 使用 Trainer 进行训练
    trainer = POITrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_module['data_collator'],
        train_dataset=data_module['train_dataset'],
    )

    def check_weights_and_grads(trainer):
        """检查 linear_mapping_layer 在每一步的权重和梯度更新。"""
        for name, param in trainer.model.named_parameters():
            if "linear_mapping_layer" in name and param.requires_grad:
                grad_sample = param.grad[0][:5].float().cpu().numpy() if param.grad is not None else "No grad computed"
                weight_sample = param.data[0][:5].float().cpu().numpy()  # 转换为 Float32
                print(f"{name} - Grad sample: {grad_sample}")
                print(f"{name} - Weight sample: {weight_sample}")

    # 自定义回调函数，在日志记录阶段检查梯度和权重
    class CheckWeightsAndGradsCallback(TrainerCallback):
        def on_log(self, args, state, control, model=None, **kwargs):
            check_weights_and_grads(trainer)

    # 添加自定义回调到 Trainer
    trainer.add_callback(CheckWeightsAndGradsCallback())

    print("Training starts...")
    print(f"POI embeddings shape: {poi_embeddings.shape}")
    print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
    print(f"First batch input ids: {next(iter(data_module['train_dataset']))['input_ids']}")

    # 开始训练
    trainer.train(resume_from_checkpoint=False)

    torch.cuda.empty_cache()

    trainer.save_state()
    # 保存模型和 LoRA 微调后的 linear_mapping_layer

    trainer.save_model(output_dir=training_args.output_dir)
    lora_layer_path = os.path.join(training_args.output_dir, "linear_mapping_layer_lora.pt")
    torch.save({k: v for k, v in model.state_dict().items() if "lora_" in k}, lora_layer_path)

    # 调用此函数保存已合并的 linear_mapping_layer
    save_combined_linear_mapping_layer(model, training_args.output_dir, alpha=training_args.lora_alpha, r=training_args.lora_r)



if __name__ == "__main__":
    train()