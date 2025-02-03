# finetune_FusionGeoEncoder.py
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
import random
import pandas as pd

# 替换 LLAMA 注意力
from llama_attn_replace_sft import replace_llama_attn

# ------------------------------------------------
# 1) 导入我们融合版的 FusionGeoEncoder4096
#    注意 fusion_geo_encoder.py 放在同级或 geo_model 文件夹下
#    例如 geo_model/fusion_geo_encoder.py 里定义了 FusionGeoEncoder4096
# ------------------------------------------------
from geo_model.fusion_geo_encoder import FusionGeoEncoder4096 as FusionGeoEncoder

# ------------------------------------------------
# 设置随机种子，确保可复现
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    # 虽然不再直接用 pre_gps_encoder_path，但保留字段防止报错

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "最大序列长度。"},
    )
    use_flash_attn: bool = field(default=True)
    use_full_attn: bool = field(default=False)
    low_rank_training: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=8)
    deepspeed: str = field(default="ds_configs/stage2.json")
    lora_r: int = field(default=8)
    lora_alpha: float = field(default=32)
    save_total_limit: int = field(default=3)

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
        super().__init__()
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
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def load_gps_mapping(gps_mapping_path):
    """从 CSV 文件加载 GPS 映射。"""
    gps_mapping_df = pd.read_csv(gps_mapping_path)
    gps_mapping = gps_mapping_df.set_index('GPS_id').T.to_dict('list')
    return gps_mapping

# ---------------------------
# Hook 函数：将 GPS token 替换为我们 FusionGeoEncoder 输出(4096维)
# ---------------------------
def embedding_hook(module, inputs, output):
    input_ids = inputs[0].to(module.weight.device)
    gps_token_start_id = module.gps_token_start_id

    is_gps_token = input_ids >= gps_token_start_id
    token_embedding = output.clone()

    gps_token_ids = input_ids[is_gps_token] - gps_token_start_id
    if gps_token_ids.size(0) > 0:
        # 获取对应的 GPS ids
        gps_ids = [module.gps_id_list[token_id] for token_id in gps_token_ids.cpu().numpy()]
        # 获取对应的经纬度
        gps_coordinates = [module.gps_mapping[gps_id] for gps_id in gps_ids]
        # 调用 FusionGeoEncoder forward -> [1,4096]
        geo_embeddings = []
        for lat, lng in gps_coordinates:
            embedding_4096 = module.geo_encoder(latitude=lat, longitude=lng)  # [1,4096]
            geo_embeddings.append(embedding_4096.squeeze(0))
        geo_embeddings = torch.stack(geo_embeddings).to(module.weight.device)  # [num_gps_tokens, 4096]

        token_embedding[is_gps_token] = geo_embeddings.to(token_embedding.dtype)
    return token_embedding

def apply_embedding_hook(model, gps_mapping, gps_id_list, gps_token_start_id, geo_encoder):
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

# 自定义 Trainer 类，用于保存合并LoRA权重后的 FusionGeoEncoder
class POITrainer(transformers.Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)

        checkpoint_folder = f"{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        # 合并并保存FusionGeoEncoder
        geo_encoder_copy = copy.deepcopy(model.geo_encoder)
        geo_encoder_copy = geo_encoder_copy.merge_and_unload()
        geo_encoder_save_path = os.path.join(output_dir, "fusion_geo_encoder_merged.pth")
        torch.save(geo_encoder_copy.state_dict(), geo_encoder_save_path)
        print(f"已将合并了 LoRA 权重的 FusionGeoEncoder 保存到 {geo_encoder_save_path}")

        # 仅保留最近N个checkpoint
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
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1. 打印配置信息
    print("模型参数：", model_args)
    print("数据参数：", data_args)
    print("训练参数：", training_args)

    # 2. 替换注意力机制
    replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # 3. 加载模型配置
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # RoPE 缩放因子
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # 4. 加载预训练基模型: 4-bit量化
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

    # 5. 如果要对大模型本身也做LoRA
    if training_args.low_rank_training:
        targets = ["q_proj","k_proj","v_proj","o_proj"]
        lora_config_main = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config_main)

    # 6. 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # 7. 加载GPS映射，并给GPS token分配ID
    gps_mapping = load_gps_mapping(data_args.gps_mapping_path)
    num_gps_tokens = len(gps_mapping)
    max_token_id = max(tokenizer.get_vocab().values())
    gps_token_start_id = max_token_id + 1

    gps_ids = list(gps_mapping.keys())
    # 根据数据集名称确定GPS token的文本
    if "nyc" in data_args.data_path.lower():
        gps_tokens = [f"<GPS {i}>" for i in range(0, num_gps_tokens)]
    elif "ca" in data_args.data_path.lower() or "tky" in data_args.data_path.lower():
        gps_tokens = [f"<GPS {i}>" for i in range(1, num_gps_tokens + 1)]
    else:
        raise ValueError("Unsupported dataset name. Please use 'nyc', 'ca', or 'tky'.")

    # 增量添加GPS tokens
    special_tokens_dict = {"additional_special_tokens": gps_tokens}
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    check_token_ids(tokenizer, [gps_token_start_id, gps_token_start_id+1, gps_token_start_id+2])

    # 8. 构建 FusionGeoEncoder4096，并打印其结构
    fusion_geo_encoder = FusionGeoEncoder(
        gps_embed_dim=128,       # n-gram embedding维度
        quadkey_length=25,
        n=6,
        vocab_size=4096,
        fourier_output_dim=128,  # Fourier部分输出128
        final_dim=4096           # 最终投影到4096
    ).to(device)

    print("\n== FusionGeoEncoder 子模块结构 ==")
    for name, module in fusion_geo_encoder.named_modules():
        print(name, ":", module)
    print("================================\n")

    # 将 fusion_geo_encoder 附加到大模型上
    model.geo_encoder = fusion_geo_encoder

    # 9. 注册hook：当遇到GPS token就用fusion_geo_encoder替换嵌入
    apply_embedding_hook(model, gps_mapping, gps_ids, gps_token_start_id, fusion_geo_encoder)

    # 10. 冻结除 LoRA & fusion_geo_encoder 之外的大模型参数
    for name, param in model.named_parameters():
        if ('lora_' in name) or ('geo_encoder' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 11. 对 fusion_geo_encoder 注入 LoRA
    if training_args.low_rank_training:
        # 初步给出一个通配式 target_modules，可以之后根据打印结构再微调
        target_modules = [
            r"gps_embed_model.gps_embedding",
            # gps_encoder: self-attn Q/K/V, out_proj, linear1, linear2
            r"gps_encoder.transformer_encoder.layers.\d+.self_attn.in_proj_weight",
            r"gps_encoder.transformer_encoder.layers.\d+.self_attn.out_proj",
            r"gps_encoder.transformer_encoder.layers.\d+.linear1",
            r"gps_encoder.transformer_encoder.layers.\d+.linear2",
            # Fourier
            r"fourier_encoder.pos_encoder.Wr",
            r"fourier_encoder.pos_encoder.mlp.*",
            # final layer
            r"final_linear"
        ]
        lora_config_geo = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        model.geo_encoder = get_peft_model(model.geo_encoder, lora_config_geo)

    print("以下参数将被训练：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(" -", name)

    # 12. 构建数据
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    trainer = POITrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_module['data_collator'],
        train_dataset=data_module['train_dataset'],
    )

    print("开始训练...")
    trainer.train(resume_from_checkpoint=False)

    torch.cuda.empty_cache()

    trainer.save_state()
    # 13. 最后手动合并LoRA权重并保存
    geo_encoder_copy = copy.deepcopy(model.geo_encoder)
    geo_encoder_copy = geo_encoder_copy.merge_and_unload()
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    geo_encoder_save_path = os.path.join(output_dir, "fusion_geo_encoder_final_merged.pth")
    torch.save(geo_encoder_copy.state_dict(), geo_encoder_save_path)
    print(f"已将最终合并了 LoRA 权重的 FusionGeoEncoder 保存到 {geo_encoder_save_path}")

if __name__ == "__main__":
    train()