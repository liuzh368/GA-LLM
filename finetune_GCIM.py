# training_script.py
import io
import os
import copy
import math
import json
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
from llama_attn_replace_sft import replace_llama_attn  # 替换 LLAMA 注意力机制

from transformers import TrainerCallback
import random

# 导入 GeoEncoder
from geo_model.geo_encoder import GeoEncoder

import pandas as pd

# 设置随机种子，确保可复现性
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确保每次使用相同的卷积算法
    torch.backends.cudnn.benchmark = False

# 在代码的开头调用 set_seed 函数
set_seed(42)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
os.environ["WANDB_DISABLED"] = "true"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """从 .json 文件加载数据为字典。"""
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
    data_path: str = field(default=None, metadata={"help": "训练数据的路径。"})
    gps_mapping_path: str = field(default=None, metadata={"help": "GPS 映射 CSV 文件的路径。"})
    pre_gps_encoder_path: str = field(default=None, metadata={"help": "Path to the trained pre_GeoEncoder model."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "最大序列长度。序列将被右侧填充（可能被截断）。"},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "是否在训练中使用 flash attention。"},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "是否在训练中使用完全注意力。"},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "是否使用低秩适应（LoRA）进行训练。"},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "在执行反向/更新之前累积的更新步骤数。"},
    )
    deepspeed: str = field(
        default="ds_configs/stage2.json",
        metadata={"help": "DeepSpeed 配置文件的路径。"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA 低秩矩阵近似的秩。"}
    )
    lora_alpha: float = field(
        default=32,
        metadata={"help": "用于合并 LoRA 权重的 Alpha 缩放因子。"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "限制检查点的总数量。在 output_dir 中删除较旧的检查点。默认为 3。"}
    )

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel
):
    """调整 tokenizer 和嵌入层的大小。"""
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
    """对字符串列表进行分词。"""
    tokenizer.truncation_side = "left"
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
    """通过分词预处理数据。"""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX  # 将 `question` 部分的损失屏蔽
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """用于有监督微调的数据集。"""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("正在加载数据...")
        list_data_dict = jload(data_path)

        logging.warning("正在格式化输入...")
        sources = [example["question"] for example in list_data_dict]
        targets = [f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]
        logging.warning("正在分词输入... 这可能需要一些时间...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """为有监督微调整理示例。"""

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
    """为有监督微调创建数据集和整理器。"""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def load_gps_mapping(gps_mapping_path):
    """从 CSV 文件加载 GPS 映射。"""
    gps_mapping_df = pd.read_csv(gps_mapping_path)
    gps_mapping = gps_mapping_df.set_index('GPS_id').T.to_dict('list')
    return gps_mapping

def embedding_hook(module, inputs, output):
    """钩子函数，动态使用 GeoEncoder 将 GPS 嵌入映射到模型的嵌入空间。"""
    input_ids = inputs[0].to(module.weight.device)
    gps_token_start_id = module.gps_token_start_id

    is_gps_token = input_ids >= gps_token_start_id
    token_embedding = output.clone()

    gps_token_ids = input_ids[is_gps_token] - gps_token_start_id
    if gps_token_ids.size(0) > 0:
        # 获取对应的 GPS ids
        gps_ids = [module.gps_id_list[token_id] for token_id in gps_token_ids.cpu().numpy()]
        # 获取对应的经纬度坐标
        gps_coordinates = [module.gps_mapping[gps_id] for gps_id in gps_ids]
        # 使用 GeoEncoder 生成嵌入
        geo_embeddings = []
        for lat, lng in gps_coordinates:
            embedding = module.geo_encoder.generate_embedding(lat, lng)
            geo_embeddings.append(embedding.squeeze(0))
        geo_embeddings = torch.stack(geo_embeddings).to(module.weight.device)  # 形状：[num_gps_tokens, embed_dim]
        # 替换对应位置的嵌入
        token_embedding[is_gps_token] = geo_embeddings.to(token_embedding.dtype)
    return token_embedding

def apply_embedding_hook(model, gps_mapping, gps_id_list, gps_token_start_id, geo_encoder):
    """给模型的嵌入层添加钩子函数。"""
    embedding_layer = model.get_input_embeddings()
    embedding_layer.gps_mapping = gps_mapping
    embedding_layer.gps_id_list = gps_id_list
    embedding_layer.gps_token_start_id = gps_token_start_id
    embedding_layer.geo_encoder = geo_encoder
    embedding_layer.register_forward_hook(embedding_hook)

def check_token_ids(tokenizer, ids_to_check):
    print("检查 Token ID 映射：")
    for token_id in ids_to_check:
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"Token ID {token_id}: {token}")

# 自定义 Trainer 类
class POITrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        # 调用原始的 _save_checkpoint 保存检查点
        super()._save_checkpoint(model, trial, metrics)

        # 保存合并了 LoRA 权重的 GeoEncoder
        checkpoint_folder = f"{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        # 创建 GeoEncoder 的副本
        geo_encoder_copy = copy.deepcopy(model.geo_encoder)

        # 合并 LoRA 权重到副本中
        geo_encoder_copy = geo_encoder_copy.merge_and_unload()

        # 保存合并后的 GeoEncoder
        geo_encoder_save_path = os.path.join(output_dir, "geo_encoder_merged.pth")
        torch.save(geo_encoder_copy.state_dict(), geo_encoder_save_path)
        print(f"已将合并了 LoRA 权重的 GeoEncoder 保存到 {geo_encoder_save_path}")

        # 仅保留最近的 3 个检查点
        checkpoints = [os.path.join(self.args.output_dir, d) for d in os.listdir(self.args.output_dir) if d.isdigit()]
        checkpoints = sorted(checkpoints, key=lambda x: int(os.path.basename(x)))
        if len(checkpoints) > self.args.save_total_limit:
            checkpoints_to_delete = checkpoints[:-self.args.save_total_limit]
            for checkpoint in checkpoints_to_delete:
                print(f"正在删除旧的检查点：{checkpoint}")
                try:
                    os.system(f"rm -rf {checkpoint}")
                except Exception as e:
                    print(f"删除检查点 {checkpoint} 时出错：{e}")

def train():
    # 解析命令行参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 输出所有参数配置
    print("模型参数：")
    print(f"  模型名称或路径：{model_args.model_name_or_path}")
    print(f"  模型类型：{model_args.model_type}")

    print("\n数据参数：")
    print(f"  数据路径：{data_args.data_path}")
    print(f"  GPS 映射路径：{data_args.gps_mapping_path}")

    print("\n训练参数：")
    print(f"  缓存目录：{training_args.cache_dir}")
    print(f"  优化器：{training_args.optim}")
    print(f"  模型最大长度：{training_args.model_max_length}")
    print(f"  使用 Flash Attention：{training_args.use_flash_attn}")
    print(f"  使用全注意力：{training_args.use_full_attn}")
    print(f"  低秩训练：{training_args.low_rank_training}")
    print(f"  梯度累积步骤数：{training_args.gradient_accumulation_steps}")
    print(f"  DeepSpeed 配置：{training_args.deepspeed}")
    print(f"  LoRA Alpha：{training_args.lora_alpha}")
    print(f"  保存检查点总数限制：{training_args.save_total_limit}")

    # 替换 LLAMA 模型的注意力机制
    replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # 加载模型的配置文件
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    # 获取模型的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置 RoPE 缩放因子
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # 加载模型，使用 4 位量化（QLoRA）
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

    # 如果使用低秩训练，在主模型上应用 LoRA
    if training_args.low_rank_training:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config_main = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config_main)

    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # 加载 GPS 映射
    gps_mapping = load_gps_mapping(data_args.gps_mapping_path)
    num_gps_tokens = len(gps_mapping)

    # 获取当前最大 Token ID
    max_token_id = max(tokenizer.get_vocab().values())

    # 设置 GPS tokens 的起始 ID
    gps_token_start_id = max_token_id + 1

    # 创建 GPS tokens 列表
    gps_ids = list(gps_mapping.keys())
    # 根据数据集名称调整 gps_tokens 的生成
    if "nyc" in data_args.data_path.lower():
        gps_tokens = [f"<GPS {i}>" for i in range(0, num_gps_tokens)]
    elif "ca" in data_args.data_path.lower() or "tky" in data_args.data_path.lower():
        gps_tokens = [f"<GPS {i}>" for i in range(1, num_gps_tokens + 1)]
    else:
        raise ValueError("Unsupported dataset name. Please use 'nyc', 'ca', or 'tky'.")

    # 将 GPS tokens 添加到 tokenizer 的词汇表中
    special_tokens_dict = {"additional_special_tokens": gps_tokens}
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    # 检查 Token IDs
    check_token_ids(tokenizer, [gps_token_start_id, gps_token_start_id + 1, gps_token_start_id + 2])

    # 获取模型的嵌入维度
    model_embedding_dim = model.get_input_embeddings().weight.shape[1]

    # 初始化 GeoEncoder，设置输出维度与模型的嵌入维度匹配
    geo_encoder = GeoEncoder(gps_encoder_path=data_args.pre_gps_encoder_path, gps_embed_dim=model_embedding_dim)

    # 将 GeoEncoder 添加到模型中
    model.geo_encoder = geo_encoder
    # 将 geo_encoder 移动到模型所在的设备
    model.geo_encoder.to(device)

    # 应用嵌入钩子以处理 GPS token 嵌入
    apply_embedding_hook(model, gps_mapping, gps_ids, gps_token_start_id, geo_encoder)

    # 冻结除 LoRA 参数和 GeoEncoder 参数之外的所有模型参数
    for name, param in model.named_parameters():
        if 'lora_' in name or 'geo_encoder' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 对 GeoEncoder 启用 LoRA
    if training_args.low_rank_training:
        # 获取 geo_encoder 中的模块名称
        target_modules = [
            r"gps_embedding",            # GPSEmbeddings 中的 nn.Embedding 层
            r".*self_attn.out_proj.*",   # MultiheadAttention 中的 out_proj 层
            r".*linear1.*",              # TransformerEncoderLayer 中的 linear1 层
            r".*linear2.*"               # TransformerEncoderLayer 中的 linear2 层
        ]

        lora_config_geo = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",  # 使用 FEATURE_EXTRACTION
        )

        model.geo_encoder = get_peft_model(model.geo_encoder, lora_config_geo)

    # 验证可训练的参数
    print("以下参数将被训练：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # 创建数据模块
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # 禁用缓存并启用梯度检查点
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 使用自定义的 POITrainer
    trainer = POITrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_module['data_collator'],
        train_dataset=data_module['train_dataset'],
    )

    print("开始训练...")
    print(f"Tokenizer 特殊 tokens: {tokenizer.special_tokens_map}")

    # 开始训练
    trainer.train(resume_from_checkpoint=False)

    torch.cuda.empty_cache()

    trainer.save_state()
    # 仅保存合并了 LoRA 权重的 GeoEncoder
    geo_encoder_copy = copy.deepcopy(model.geo_encoder)
    geo_encoder_copy = geo_encoder_copy.merge_and_unload()
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    geo_encoder_save_path = os.path.join(output_dir, "geo_encoder_final_merged.pth")
    torch.save(geo_encoder_copy.state_dict(), geo_encoder_save_path)
    print(f"已将最终合并了 LoRA 权重的 GeoEncoder 保存到 {geo_encoder_save_path}")

if __name__ == "__main__":
    train()