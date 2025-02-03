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
import pandas as pd

# Import the GeoEncoder
from geo_model.geo_encoder import GeoEncoder
from llama_attn_replace_sft import replace_llama_attn  # Replace LLAMA attention mechanism

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
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
    gps_mapping_path: str = field(default=None, metadata={"help": "Path to the GPS mapping CSV file."})
    geoencoder_path: str = field(default=None, metadata={"help": "Path to the trained GeoEncoder model."})

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
    # 设置 tokenizer 的截断方向
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
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX  # Mask the question part for loss computation
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

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
        device = poi_embedding_layer.weight.device  # Get the device of the poi_embedding_layer
        poi_embeddings = poi_embeddings.to(device)
        poi_embeddings = linear_mapping_layer(poi_embeddings).to(device)
        poi_embedding_layer.weight.data = poi_embeddings
    print("POI embeddings successfully initialized.")

def load_gps_mapping(gps_mapping_path):
    """Load GPS mapping from CSV."""
    gps_mapping_df = pd.read_csv(gps_mapping_path)
    gps_mapping = gps_mapping_df.set_index('GPS_id').T.to_dict('list')
    return gps_mapping

def embedding_hook(module, inputs, output):
    """Embedding hook to handle both POI and GPS tokens."""
    input_ids = inputs[0].to(module.weight.device)
    token_embedding = output.clone()

    # Access necessary components from the module
    poi_embedding_layer = module.poi_embedding_layer.weight
    poi_token_start_id = module.poi_token_start_id
    gps_token_start_id = module.gps_token_start_id
    gps_mapping = module.gps_mapping
    gps_id_list = module.gps_id_list
    geo_encoder = module.geo_encoder

    # Define the token ranges for POI and GPS
    is_poi_token = (input_ids >= poi_token_start_id) & (input_ids < gps_token_start_id)  # Using the reference line
    is_gps_token = input_ids >= gps_token_start_id

    # Debug: Print token IDs and start IDs
    print(f"POI token start ID: {poi_token_start_id}")
    print(f"GPS token start ID: {gps_token_start_id}")
    print(f"Input IDs: {input_ids}")

    # Handle POI tokens
    poi_token_ids = input_ids[is_poi_token] - poi_token_start_id
    if poi_token_ids.size(0) > 0:
        print(f"POI token IDs: {poi_token_ids}")
        poi_token_ids = poi_token_ids.to(poi_embedding_layer.device)
        if poi_token_ids.max().item() < poi_embedding_layer.size(0):  # Ensure within bounds
            poi_token_embedding = poi_embedding_layer[poi_token_ids]
            token_embedding[is_poi_token] = poi_token_embedding.to(token_embedding.dtype)
        else:
            print("Error: POI token ID exceeds embedding layer size!")

    # Handle GPS tokens
    gps_token_ids = input_ids[is_gps_token] - gps_token_start_id
    if gps_token_ids.size(0) > 0:
        print(f"GPS token IDs: {gps_token_ids}")
        gps_ids = [gps_id_list[token_id] for token_id in gps_token_ids.cpu().numpy()]
        gps_coordinates = [gps_mapping[gps_id] for gps_id in gps_ids]

        # Generate geo_embeddings and check shape
        geo_embeddings = [geo_encoder.generate_embedding(lat, lng).squeeze(0) for lat, lng in gps_coordinates]
        geo_embeddings = torch.stack(geo_embeddings).to(module.weight.device)
        token_embedding[is_gps_token] = geo_embeddings.to(token_embedding.dtype)

    return token_embedding

def apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id, gps_mapping, gps_id_list, gps_token_start_id, geo_encoder):
    """Apply embedding hook to handle POI and GPS tokens."""
    embedding_layer = model.get_input_embeddings()
    embedding_layer.poi_embedding_layer = poi_embedding_layer
    embedding_layer.poi_token_start_id = poi_token_start_id
    embedding_layer.gps_mapping = gps_mapping
    embedding_layer.gps_id_list = gps_id_list
    embedding_layer.gps_token_start_id = gps_token_start_id
    embedding_layer.geo_encoder = geo_encoder
    embedding_layer.register_forward_hook(embedding_hook)

def load_projector(projector_path, input_dim=128, output_dim=4096):
    """Load the pre-trained projector (linear mapping layer)."""
    linear_mapping_layer = nn.Linear(input_dim, output_dim)
    state_dict = torch.load(projector_path)
    linear_mapping_layer.load_state_dict(state_dict)
    return linear_mapping_layer


# *** Start of Modifications ***

def remove_lora_keys(state_dict):
    """
    Remove keys containing 'lora_' from the state_dict.
    """
    keys_to_remove = [key for key in state_dict.keys() if 'lora_' in key]
    for key in keys_to_remove:
        del state_dict[key]
    return state_dict

def fix_key_mismatch(state_dict):
    """
    修改键名以匹配 GeoEncoder 的参数。
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # 替换键名中的 "base_layer" 为 "weight"
        if "gps_embedding.base_layer" in key:
            new_key = key.replace("gps_embedding.base_layer", "gps_embedding")
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def train():
    # Parse command-line arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Replace LLAMA attention mechanism
    replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # Load model configuration
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    # Set RoPE scaling factor
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model, using 4-bit quantization (QLoRA)
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
    )

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and initialize POI embeddings
    poi_embeddings = load_poi_embeddings(data_args.poi_embedding_path)
    linear_mapping_layer = load_projector(data_args.projector_path)
    max_token_id = max(tokenizer.get_vocab().values())
    poi_token_start_id = max_token_id + 1
    num_poi_tokens = len(poi_embeddings)
    model_embedding_dim = model.get_input_embeddings().weight.shape[1]
    poi_embedding_layer = nn.Embedding(num_embeddings=num_poi_tokens, embedding_dim=model_embedding_dim)
    initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, linear_mapping_layer)

    # Load GPS mapping and initialize GeoEncoder
    gps_mapping = load_gps_mapping(data_args.gps_mapping_path)
    gps_ids = list(gps_mapping.keys())
    num_gps_tokens = len(gps_ids)
    gps_token_start_id = poi_token_start_id + num_poi_tokens

    def strip_prefix_from_state_dict(state_dict, prefix):
        """
        去除 state_dict 中的指定前缀。

        Args:
            state_dict (dict): 模型的 state_dict。
            prefix (str): 需要移除的前缀。

        Returns:
            dict: 处理后的 state_dict。
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]  # 去掉前缀
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    geo_encoder = GeoEncoder(
        gps_embed_dim=4096,
        num_gps=4096,
        quadkey_length=25,
        n=6
    )
    # 加载权重并去除前缀
    geoencoder_state_dict = torch.load(data_args.geoencoder_path, map_location=device)
    geoencoder_state_dict = strip_prefix_from_state_dict(geoencoder_state_dict, "base_model.model.")

    # 修复键名不匹配问题
    geoencoder_state_dict = fix_key_mismatch(geoencoder_state_dict)

    # *** Apply the function to remove LoRA keys ***
    geoencoder_state_dict = remove_lora_keys(geoencoder_state_dict)

    # 加载到 GeoEncoder 模型中
    geo_encoder.load_state_dict(geoencoder_state_dict)
    geo_encoder.to(device)
    geo_encoder.eval()

    # 根据数据集名称决定 POI tokens 的生成方式
    if "nyc" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(0, num_poi_tokens)]
        gps_tokens = [f"<GPS {i}>" for i in range(num_gps_tokens)]
    elif "ca" in data_args.data_path.lower() or "tky" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(1, num_poi_tokens + 1)]
        gps_tokens = [f"<GPS {i}>" for i in range(1, num_gps_tokens + 1)]
    else:
        raise ValueError("Unsupported dataset name. Please use 'nyc', 'ca', or 'tky'.")


    special_tokens_dict = {"additional_special_tokens": poi_tokens + gps_tokens}
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    # Apply embedding hook to handle POI and GPS tokens
    apply_embedding_hook(
        model,
        poi_embedding_layer,
        poi_token_start_id,
        gps_mapping,
        gps_ids,
        gps_token_start_id,
        geo_encoder
    )

    # Freeze model parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Enable LoRA for low-rank adaptation
    if training_args.low_rank_training:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Enable additional trainable parameters
        trainable_param_names = training_args.trainable_params.split(",")
        for name, param in model.named_parameters():
            if any(tp in name for tp in trainable_param_names):
                param.requires_grad = True

    # Prepare data module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # 禁用缓存，启用梯度检查点
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_module['data_collator'],
        train_dataset=data_module['train_dataset'],
    )

    # Start training
    trainer.train(resume_from_checkpoint=False)

    torch.cuda.empty_cache()

    # Save model and state
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()