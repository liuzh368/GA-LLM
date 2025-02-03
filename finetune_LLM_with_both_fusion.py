# finetune_LLM_with_both_fusion_fixed.py
# ---------------------------------------------------
# 参考你的第三份代码 + geo_model/fusion_geo_encoder.py，
# 修复 "FusionGeoEncoder4096 object has no attribute 'generate_embedding'" 问题。
# 其余逻辑保持一致，可直接运行。
# ---------------------------------------------------

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
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
import pandas as pd
import random

# ---- 解决 "FusionGeoEncoder4096 没有 generate_embedding" 的关键：我们直接把类定义在同一个脚本里，并添加这个方法 ----

EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
MinLongitude = -180
MaxLongitude = 180

def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)

def map_size(levelOfDetail):
    return 256 << levelOfDetail

def latlng2pxy(latitude, longitude, levelOfDetail):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, MinLongitude, MaxLongitude)

    x = (longitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    ms = map_size(levelOfDetail)
    pixelX = int(clip(x * ms + 0.5, 0, ms - 1))
    pixelY = int(clip(y * ms + 0.5, 0, ms - 1))
    return pixelX, pixelY

def pxy2txy(pixelX, pixelY):
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY

def txy2quadkey(tileX, tileY, levelOfDetail):
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))
    return ''.join(quadKey)

def latlng2quadkey(lat, lng, level=25):
    pixelX, pixelY = latlng2pxy(lat, lng, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY, level)

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, G, M, F_dim, H_dim, D, gamma=10.0):
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        device = self.Wr.weight.device
        dtype  = self.Wr.weight.dtype
        x = x.to(device=device, dtype=dtype)

        N, G, M = x.shape
        projected = self.Wr(x)                # [N, G, F_dim//2]
        cosines = torch.cos(projected)
        sines   = torch.sin(projected)
        F = (1 / math.sqrt(self.F_dim)) * torch.cat([cosines, sines], dim=-1)  # [N, G, F_dim]

        Y = self.mlp(F)                       # [N, G, D//G]
        PEx = Y.reshape(N, self.D)
        return PEx

class GPSPositionalEncoder(nn.Module):
    def __init__(self, M=25, F_dim=256, H_dim=128, D=128, gamma=10.0):
        super().__init__()
        self.G = 1
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        self.pos_encoder = LearnableFourierPositionalEncoding(
            G=self.G,
            M=self.M,
            F_dim=self.F_dim,
            H_dim=self.H_dim,
            D=self.D,
            gamma=self.gamma
        )

    def forward(self, latitude, longitude):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        # 生成QuadKey(25维)
        quadkey = latlng2quadkey(latitude, longitude, level=self.M)
        quadkey_digits = [int(c) for c in quadkey]  # len=25
        x = torch.tensor(quadkey_digits, dtype=dtype).unsqueeze(0).unsqueeze(0).to(device)

        pex = self.pos_encoder(x)  # [1, D]
        return pex

class GPSEmbeddings(nn.Module):
    def __init__(self, num_gps, embedding_dim):
        super().__init__()
        self.gps_embedding = nn.Embedding(num_embeddings=num_gps, embedding_dim=embedding_dim)

    def forward(self, gps_idx):
        gps_idx = gps_idx.to(self.gps_embedding.weight.device)
        embed = self.gps_embedding(gps_idx)
        return embed

class GPSEncoder(nn.Module):
    def __init__(self, embed_size=128, nhead=1, nhid=256, nlayers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.embed_size = embed_size
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, src):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        src = src.to(device=device, dtype=dtype)

        src = src * math.sqrt(self.embed_size)
        x = self.transformer_encoder(src)  # [batch, seq_len, embed_size]
        x = torch.mean(x, dim=1)           # [batch, embed_size]
        return self.norm(x)

class FusionGeoEncoder4096(nn.Module):
    """
    - n-gram + Embedding(128) + Transformer => [1,128]
    - QuadKey learnable Fourier => [1,128]
    - concat => [1,256] => Linear => [1,4096]
    """
    def __init__(self,
                 gps_embed_dim=128,
                 quadkey_length=25,
                 n=6,
                 vocab_size=4096,
                 fourier_output_dim=128,
                 final_dim=4096,
                 dropout=0.1):
        super().__init__()
        self.gps_embed_dim = gps_embed_dim
        self.quadkey_length = quadkey_length
        self.n = n

        from itertools import product
        chars = ['0','1','2','3']
        all_permutations = [''.join(p) for p in product(chars, repeat=self.n)]
        self.permutations_dict = dict(zip(all_permutations, range(len(all_permutations))))

        self.gps_embed_model = GPSEmbeddings(num_gps=vocab_size, embedding_dim=gps_embed_dim)
        self.gps_encoder = GPSEncoder(
            embed_size=gps_embed_dim,
            nhead=1,
            nhid=2*gps_embed_dim,
            nlayers=2,
            dropout=dropout
        )
        self.fourier_encoder = GPSPositionalEncoder(
            M=self.quadkey_length,
            F_dim=256,
            H_dim=128,
            D=fourier_output_dim,
            gamma=10.0
        )

        self.final_linear = nn.Linear(gps_embed_dim + fourier_output_dim, final_dim)

    def forward(self, latitude, longitude):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        # Step A: n-gram => [1, seq_len]
        quadkey = latlng2quadkey(latitude, longitude, self.quadkey_length)
        # 取 n-grams
        import nltk
        from nltk import ngrams
        quadkey_ngrams = [''.join(x) for x in ngrams(quadkey, self.n)]
        qk_idx = [self.permutations_dict.get(ng, 0) for ng in quadkey_ngrams]
        qk_idx_tensor = torch.tensor(qk_idx, dtype=torch.long).unsqueeze(0).to(device)

        token_embeddings = self.gps_embed_model(qk_idx_tensor)  # [1, seq_len, 128]
        token_embeddings = token_embeddings.to(dtype=dtype)

        emb_128 = self.gps_encoder(token_embeddings)  # [1,128]

        # Step B: Fourier => [1,128]
        fourier_128 = self.fourier_encoder(latitude, longitude)
        fourier_128 = fourier_128.to(device=device, dtype=dtype)

        # concat => [1,256]
        fused = torch.cat([emb_128, fourier_128], dim=-1)  # [1,256]
        fused = fused.to(device=device, dtype=dtype)

        final_4096 = self.final_linear(fused)  # [1,4096]
        return final_4096

    # ---- 新增 generate_embedding 方法，以兼容 embedding_hook 中的调用 ----
    def generate_embedding(self, lat, lng):
        return self.forward(lat, lng)

# -----------------------------------------------------------------------
# 设置随机种子
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
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")
    linear_mapping_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the pre-trained projector (linear layer)."}
    )
    fusion_geoencoder_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the pre-trained FusionGeoEncoder (merged.pth)."}
    )
    peft_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to the directory containing LoRA fine-tuned parameters (optional)."}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    poi_embedding_path: str = field(default=None, metadata={"help": "Path to the POI embeddings (.npy file)."})
    gps_mapping_path: str = field(default=None, metadata={"help": "Path to the GPS mapping CSV file."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="output", metadata={"help": "Output directory for the model."})
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention during training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use full attention during training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether to use low-rank adaptation (LoRA) during training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if using low-rank training."},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of update steps to accumulate before performing a backward/update pass."},
    )
    deepspeed: str = field(
        default=None,
        metadata={"help": "Path to the DeepSpeed config file, if using DeepSpeed."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank parameter."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "LoRA alpha parameter."}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir."}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel
):
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
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")
        sources = [example["question"] for example in list_data_dict]
        targets = [f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]
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

def load_poi_embeddings(poi_embedding_path):
    poi_embeddings = np.load(poi_embedding_path)
    return torch.tensor(poi_embeddings)

def initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, linear_mapping_layer, device):
    with torch.no_grad():
        poi_embedding_layer.to(device)
        linear_mapping_layer.to(device)
        poi_embeddings = poi_embeddings.to(device)
        poi_embeddings = linear_mapping_layer(poi_embeddings).to(device)
        poi_embedding_layer.weight.data = poi_embeddings
    print("POI embeddings successfully initialized.")

def load_gps_mapping(gps_mapping_path):
    gps_mapping_df = pd.read_csv(gps_mapping_path)
    gps_mapping = gps_mapping_df.set_index('GPS_id').T.to_dict('list')
    return gps_mapping

def embedding_hook(module, inputs, output):
    """Embedding hook to handle both POI and GPS tokens."""
    input_ids = inputs[0].to(module.weight.device)
    token_embedding = output.clone()

    poi_embedding_layer = module.poi_embedding_layer.weight
    poi_token_start_id = module.poi_token_start_id
    gps_token_start_id = module.gps_token_start_id
    gps_mapping = module.gps_mapping
    gps_id_list = module.gps_id_list
    fusion_geo_encoder = module.fusion_geo_encoder  # 用 fusion_geo_encoder

    # 确保 fusion_geo_encoder 在正确的设备上
    if fusion_geo_encoder is not None:
        fusion_geo_encoder_device = next(fusion_geo_encoder.parameters()).device
        assert fusion_geo_encoder_device == module.weight.device, "FusionGeoEncoder 不在正确的设备上!"
    else:
        print("Warning: fusion_geo_encoder is None!")

    is_poi_token = (input_ids >= poi_token_start_id) & (input_ids < gps_token_start_id)
    is_gps_token = input_ids >= gps_token_start_id

    # 打印信息帮助调试
    print(f"POI token start ID: {poi_token_start_id}")
    print(f"GPS token start ID: {gps_token_start_id}")
    print(f"Input IDs: {input_ids}")

    # 处理 POI tokens
    poi_token_ids = input_ids[is_poi_token] - poi_token_start_id
    if poi_token_ids.size(0) > 0:
        print(f"POI token IDs: {poi_token_ids}")
        poi_token_ids = poi_token_ids.to(poi_embedding_layer.device)
        if poi_token_ids.max().item() < poi_embedding_layer.size(0):
            poi_token_embedding = poi_embedding_layer[poi_token_ids]
            token_embedding[is_poi_token] = poi_token_embedding.to(token_embedding.dtype)
        else:
            print("Error: POI token ID exceeds embedding layer size!")

    # 处理 GPS tokens
    gps_token_ids = input_ids[is_gps_token] - gps_token_start_id
    if gps_token_ids.size(0) > 0:
        print(f"GPS token IDs: {gps_token_ids}")
        gps_ids = [gps_id_list[token_id] for token_id in gps_token_ids.cpu().numpy()]
        gps_coordinates = [gps_mapping[gps_id] for gps_id in gps_ids]

        geo_embeddings = []
        for lat, lng in gps_coordinates:
            # 关键：FusionGeoEncoder4096 已经有 generate_embedding
            embedding = fusion_geo_encoder.generate_embedding(lat, lng)
            if embedding.ndim > 1:
                embedding = embedding.squeeze(0)
            geo_embeddings.append(embedding)
        if len(geo_embeddings) > 0:
            geo_embeddings = torch.stack(geo_embeddings).to(module.weight.device)
            token_embedding[is_gps_token] = geo_embeddings.to(token_embedding.dtype)

    return token_embedding

def apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id, gps_mapping, gps_id_list,
                         gps_token_start_id, fusion_geo_encoder, max_token_id):
    embedding_layer = model.get_input_embeddings()
    embedding_layer.poi_embedding_layer = poi_embedding_layer
    embedding_layer.poi_token_start_id = poi_token_start_id
    embedding_layer.gps_mapping = gps_mapping
    embedding_layer.gps_id_list = gps_id_list
    embedding_layer.gps_token_start_id = gps_token_start_id
    embedding_layer.fusion_geo_encoder = fusion_geo_encoder
    embedding_layer.register_forward_hook(embedding_hook)

def load_projector(projector_path, input_dim=128, output_dim=4096):
    linear_mapping_layer = nn.Linear(input_dim, output_dim)
    state_dict = torch.load(projector_path, map_location="cpu")
    linear_mapping_layer.load_state_dict(state_dict)
    return linear_mapping_layer

def remove_lora_keys(state_dict):
    keys_to_remove = [key for key in state_dict.keys() if 'lora_' in key]
    for key in keys_to_remove:
        del state_dict[key]
    return state_dict

def fix_key_mismatch(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "gps_embedding.base_layer" in key:
            new_key = key.replace("gps_embedding.base_layer", "gps_embed_model.gps_embedding")
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.poi_embedding_path is None or data_args.gps_mapping_path is None:
        raise ValueError("必须提供 poi_embedding_path 和 gps_mapping_path。")

    # 替换 Llama 注意力机制
    from llama_attn_replace_sft import replace_llama_attn
    replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto"
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 初始化 POI embeddings ----
    poi_embeddings = load_poi_embeddings(data_args.poi_embedding_path)
    linear_mapping_layer = load_projector(model_args.linear_mapping_path)
    linear_mapping_layer = linear_mapping_layer.to(device)

    max_token_id = max(tokenizer.get_vocab().values())
    poi_token_start_id = max_token_id + 1
    num_poi_tokens = len(poi_embeddings)
    gps_token_start_id = poi_token_start_id + num_poi_tokens

    model_embedding_dim = model.get_input_embeddings().weight.shape[1]
    poi_embedding_layer = nn.Embedding(num_embeddings=num_poi_tokens, embedding_dim=model_embedding_dim).to(device)
    initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, linear_mapping_layer, device)

    model.linear_mapping_layer = linear_mapping_layer

    # ---- FusionGeoEncoder + GPS mapping ----
    gps_mapping = load_gps_mapping(data_args.gps_mapping_path)
    gps_id_list = list(gps_mapping.keys())
    num_gps_tokens = len(gps_id_list)

    fusion_geo_encoder = FusionGeoEncoder4096(
        gps_embed_dim=128,
        quadkey_length=25,
        n=6,
        vocab_size=4096,
        fourier_output_dim=128,
        final_dim=model.config.hidden_size,
        dropout=0.1
    ).to(device)

    if model_args.fusion_geoencoder_path:
        fusion_geoencoder_state_dict = torch.load(model_args.fusion_geoencoder_path, map_location=device)
        fusion_geoencoder_state_dict = fix_key_mismatch(fusion_geoencoder_state_dict)
        fusion_geoencoder_state_dict = remove_lora_keys(fusion_geoencoder_state_dict)
        fusion_geo_encoder.load_state_dict(fusion_geoencoder_state_dict, strict=False)
        fusion_geo_encoder.eval()

    model.fusion_geo_encoder = fusion_geo_encoder

    # ---- 为 tokenizer 添加 <POI x>, <GPS x> token 并 resize ----
    if "nyc" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(num_poi_tokens)]
        gps_tokens = [f"<GPS {i}>" for i in range(num_gps_tokens)]
    elif "ca" in data_args.data_path.lower() or "tky" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(1, num_poi_tokens + 1)]
        gps_tokens = [f"<GPS {i}>" for i in range(1, num_gps_tokens + 1)]
    else:
        raise ValueError("Unsupported dataset name. Please use 'nyc', 'ca', or 'tky'.")

    special_tokens_dict = {"additional_special_tokens": poi_tokens + gps_tokens}
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    # ---- 注册 forward_hook ----
    apply_embedding_hook(
        model,
        poi_embedding_layer,
        poi_token_start_id,
        gps_mapping,
        gps_id_list,
        gps_token_start_id,
        fusion_geo_encoder,
        max_token_id
    )

    # 冻结模型参数
    for name, param in model.named_parameters():
        param.requires_grad = False

    # projector
    if hasattr(model, 'linear_mapping_layer'):
        model.linear_mapping_layer.requires_grad = True

    # fusion_geo_encoder
    for param in fusion_geo_encoder.parameters():
        param.requires_grad = True

    # ---- LoRA 训练 ----
    if training_args.low_rank_training:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "linear_mapping_layer"]
        lora_config_main = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config_main)

        trainable_param_names = training_args.trainable_params.split(",")
        for name, param in model.named_parameters():
            if any(tp in name for tp in trainable_param_names):
                param.requires_grad = True

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

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

    trainer.train(resume_from_checkpoint=False)
    torch.cuda.empty_cache()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    print(f"模型已保存到 {training_args.output_dir}")

if __name__ == "__main__":
    train()