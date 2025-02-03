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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    gps_embedding_path: str = field(default=None, metadata={"help": "Path to the GPS embeddings (.npy file)."})


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


def load_embeddings(embedding_path):
    """Load embeddings from a .npy file."""
    embeddings = np.load(embedding_path)
    return torch.tensor(embeddings)


def initialize_embeddings(embedding_layer, embeddings, device):
    """Initialize embedding layer without mapping to high dimension."""
    with torch.no_grad():
        embedding_layer.to(device)
        embeddings = embeddings.to(device)

        # Set the embeddings in the embedding layer
        embedding_layer.weight.data = embeddings
    print("Initialized embeddings.")


def embedding_hook(module, inputs, output):
    """Hook function to dynamically use poi_mapping_layer and gps_mapping_layer."""
    input_ids = inputs[0].to(module.weight.device)
    poi_token_start_id = module.poi_token_start_id
    gps_token_start_id = module.gps_token_start_id

    is_poi_token = (input_ids >= poi_token_start_id) & (input_ids < gps_token_start_id)
    is_gps_token = input_ids >= gps_token_start_id

    token_embedding = output.clone()

    if is_poi_token.any():
        poi_token_ids = input_ids[is_poi_token] - poi_token_start_id
        if poi_token_ids.max() >= module.poi_embedding_layer.num_embeddings:
            raise ValueError("POI token ID exceeds the POI embedding layer size.")
        original_poi_embedding = module.poi_embedding_layer(poi_token_ids)
        poi_token_embedding = module.poi_mapping_layer(original_poi_embedding)
        token_embedding[is_poi_token] = poi_token_embedding.to(token_embedding.dtype)

    if is_gps_token.any():
        gps_token_ids = input_ids[is_gps_token] - gps_token_start_id
        if gps_token_ids.max() >= module.gps_embedding_layer.num_embeddings:
            raise ValueError("GPS token ID exceeds the GPS embedding layer size.")
        original_gps_embedding = module.gps_embedding_layer(gps_token_ids)
        gps_token_embedding = module.gps_mapping_layer(original_gps_embedding)
        token_embedding[is_gps_token] = gps_token_embedding.to(token_embedding.dtype)

    return token_embedding


# Ensure embedding_hook applies correctly
def apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id, gps_embedding_layer, gps_token_start_id):
    """Attach hook to embedding layer for POI and GPS embeddings."""
    embedding_layer = model.get_input_embeddings()
    embedding_layer.poi_embedding_layer = poi_embedding_layer
    embedding_layer.poi_token_start_id = poi_token_start_id
    embedding_layer.gps_embedding_layer = gps_embedding_layer
    embedding_layer.gps_token_start_id = gps_token_start_id
    embedding_layer.poi_mapping_layer = model.poi_mapping_layer
    embedding_layer.gps_mapping_layer = model.gps_mapping_layer
    embedding_layer.register_forward_hook(embedding_hook)


# Check specific token IDs for mapped text
def check_token_ids(tokenizer, ids_to_check):
    print("Checking token ID mappings:")
    for token_id in ids_to_check:
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"Token ID {token_id}: {token}")


def save_combined_mapping_layer(model, output_path, alpha, r, layer_name):
    """Save the combined weights of mapping layers (poi_mapping_layer, gps_mapping_layer)."""
    with torch.no_grad():
        module = getattr(model, layer_name, None)
        if module is None:
            raise ValueError(f"{layer_name} not found in the model.")

        combined_weight_path = os.path.join(output_path, f"{layer_name}_combined.pt")
        combined_state = {}

        for i, layer in enumerate(module):
            if isinstance(layer, nn.Linear):
                # Combine LoRA weights if they exist
                if hasattr(layer, "lora_A") and hasattr(layer, "lora_B"):
                    lora_weight = (alpha / r) * (layer.lora_B["default"].weight @ layer.lora_A["default"].weight)
                    layer.weight.data += lora_weight

                # Save Linear layer weights and biases
                combined_state[f"layer_{i}_weight"] = layer.weight.data.clone()
                if layer.bias is not None:
                    combined_state[f"layer_{i}_bias"] = layer.bias.data.clone()

            elif isinstance(layer, nn.GELU):
                combined_state[f"layer_{i}_type"] = "GELU"

        # Save the full layer structure to a file
        torch.save(combined_state, combined_weight_path)
        print(f"Full {layer_name} structure saved to: {combined_weight_path}")


class POITrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        """Override the checkpoint saving method to include poi_mapping_layer and gps_mapping_layer checkpoints."""
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the model's state dict
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        print(f"Model state_dict saved to: {output_dir}/pytorch_model.bin")

        # Save the combined poi_mapping_layer and gps_mapping_layer
        save_combined_mapping_layer(
            model=self.model,
            output_path=output_dir,
            alpha=self.args.lora_alpha,
            r=self.args.lora_r,
            layer_name="poi_mapping_layer"
        )
        save_combined_mapping_layer(
            model=self.model,
            output_path=output_dir,
            alpha=self.args.lora_alpha,
            r=self.args.lora_r,
            layer_name="gps_mapping_layer"
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
    device = model.device  # 使用模型的设备
    # 加载POI和GPS embeddings
    poi_embeddings = load_embeddings(data_args.poi_embedding_path).to(device)
    gps_embeddings = load_embeddings(data_args.gps_embedding_path).to(device)

    num_poi_tokens = len(poi_embeddings)
    num_gps_tokens = len(gps_embeddings)

    # 创建专用的 POI 和 GPS embedding 层，其维度应与大模型的 token embedding 一致 (4096 维)
    model_embedding_dim = model.get_input_embeddings().weight.shape[1]
    poi_embedding_layer = nn.Embedding(num_embeddings=num_poi_tokens, embedding_dim=model_embedding_dim)
    gps_embedding_layer = nn.Embedding(num_embeddings=num_gps_tokens, embedding_dim=model_embedding_dim)



    # 创建并注册 poi_mapping_layer 和 gps_mapping_layer
    poi_mapping_layer = nn.Sequential(
        nn.Linear(128, model_embedding_dim),
        nn.GELU(),
        nn.Linear(model_embedding_dim, model_embedding_dim)
    ).to(device)
    gps_mapping_layer = nn.Sequential(
        nn.Linear(128, model_embedding_dim),
        nn.GELU(),
        nn.Linear(model_embedding_dim, model_embedding_dim)
    ).to(device)

    # 将 mapping_layers 添加到模型中作为子模块
    model.add_module('poi_mapping_layer', poi_mapping_layer)
    model.add_module('gps_mapping_layer', gps_mapping_layer)

    # 初始化 POI 和 GPS token 的嵌入
    initialize_embeddings(poi_embedding_layer, poi_embeddings, device)
    initialize_embeddings(gps_embedding_layer, gps_embeddings, device)

    # 检查当前的最大 Token ID
    max_token_id = max(tokenizer.get_vocab().values())
    print(f"Max Token ID in tokenizer: {max_token_id}")

    poi_token_start_id = max_token_id + 1  # 设定 POI token 的起始 ID
    gps_token_start_id = poi_token_start_id + num_poi_tokens  # 设定 GPS token 的起始 ID

    print(f"POI Token Start ID: {poi_token_start_id}, GPS Token Start ID: {gps_token_start_id}")

    # 添加 POI 和 GPS tokens 到 tokenizer 的词汇表中
    poi_tokens = [f"<POI {i}>" for i in range(num_poi_tokens)]
    gps_tokens = [f"<GPS {i}>" for i in range(num_gps_tokens)]
    special_tokens_dict = {"additional_special_tokens": poi_tokens + gps_tokens}
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    # 检查 token ID
    check_token_ids(tokenizer, [32000, 32001, 32002])

    # 应用钩子函数，处理 POI 和 GPS token embedding
    apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id, gps_embedding_layer, gps_token_start_id)

    # 冻结模型中的所有权重，仅微调 poi_mapping_layer 和 gps_mapping_layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.poi_mapping_layer.parameters():
        param.requires_grad = True
    for param in model.gps_mapping_layer.parameters():
        param.requires_grad = True

    # 启用 LoRA 对 mapping layers 微调
    if training_args.low_rank_training:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["poi_mapping_layer.0", "poi_mapping_layer.2", "gps_mapping_layer.0", "gps_mapping_layer.2"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

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

    trainer.train(resume_from_checkpoint=False)
    torch.cuda.empty_cache()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()