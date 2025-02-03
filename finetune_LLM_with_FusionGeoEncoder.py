# finetune_LLM_with_FusionGeoEncoder.py

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

# Import the FusionGeoEncoder
from geo_model.fusion_geo_encoder import FusionGeoEncoder4096 as FusionGeoEncoder
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
    model_name_or_path: Optional[str] = field(default="path_to_your_pretrained_LLM")
    model_type: Optional[str] = field(default="llama")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    gps_mapping_path: str = field(default=None, metadata={"help": "Path to the GPS mapping CSV file."})
    fusion_geoencoder_path: str = field(default=None, metadata={"help": "Path to the trained FusionGeoEncoder model."})

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

def load_gps_mapping(gps_mapping_path):
    """Load GPS mapping from CSV."""
    gps_mapping_df = pd.read_csv(gps_mapping_path)
    gps_mapping = gps_mapping_df.set_index('GPS_id').T.to_dict('list')
    return gps_mapping

def fusion_embedding_hook(module, inputs, output):
    """
    Embedding hook function for LLM finetune stage using FusionGeoEncoder.
    Replaces embedding for GPS tokens with the output of the pre-trained FusionGeoEncoder.
    """
    input_ids = inputs[0].to(module.weight.device)
    gps_token_start_id = module.gps_token_start_id
    gps_mapping = module.gps_mapping
    gps_id_list = module.gps_id_list
    fusion_geo_encoder = module.fusion_geo_encoder  # Instead of geo_encoder

    is_gps_token = input_ids >= gps_token_start_id
    token_embedding = output.clone()

    gps_token_ids = input_ids[is_gps_token] - gps_token_start_id
    if gps_token_ids.size(0) > 0:
        # Get lat/lng from gps_mapping
        gps_ids = [gps_id_list[token_id] for token_id in gps_token_ids.cpu().numpy()]
        gps_coordinates = [gps_mapping[gps_id] for gps_id in gps_ids]

        # Use FusionGeoEncoder to generate embedding
        fusion_embeddings = []
        for lat, lng in gps_coordinates:
            emb = fusion_geo_encoder(lat, lng)  # [1,4096]
            fusion_embeddings.append(emb.squeeze(0))

        fusion_embeddings = torch.stack(fusion_embeddings).to(module.weight.device)
        token_embedding[is_gps_token] = fusion_embeddings.to(token_embedding.dtype)

    return token_embedding

def apply_fusion_embedding_hook(model, gps_mapping, gps_id_list, gps_token_start_id, fusion_geo_encoder):
    """Attach the hook for FusionGeoEncoder-based embedding."""
    embedding_layer = model.get_input_embeddings()
    embedding_layer.gps_mapping = gps_mapping
    embedding_layer.gps_id_list = gps_id_list
    embedding_layer.gps_token_start_id = gps_token_start_id
    embedding_layer.fusion_geo_encoder = fusion_geo_encoder  # store fusion encoder
    embedding_layer.register_forward_hook(fusion_embedding_hook)

def check_token_ids(tokenizer, ids_to_check):
    print("Checking Token ID mappings:")
    for token_id in ids_to_check:
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"Token ID {token_id}: {token}")

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Replace LLAMA attention mechanism
    from llama_attn_replace_sft import replace_llama_attn
    replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    # Set RoPE scaling factor
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load LLM (4-bit quantization)
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

    # Load GPS mapping
    gps_mapping = load_gps_mapping(data_args.gps_mapping_path)
    gps_ids = list(gps_mapping.keys())
    num_gps_tokens = len(gps_ids)
    max_token_id = max(tokenizer.get_vocab().values())
    gps_token_start_id = max_token_id + 1

    # Create GPS tokens
    if "nyc" in data_args.data_path.lower():
        gps_tokens = [f"<GPS {i}>" for i in range(0, num_gps_tokens)]
    elif "ca" in data_args.data_path.lower() or "tky" in data_args.data_path.lower():
        gps_tokens = [f"<GPS {i}>" for i in range(1, num_gps_tokens + 1)]
    else:
        raise ValueError("Unsupported dataset name. Please use 'nyc', 'ca', or 'tky'.")

    # Add GPS tokens
    special_tokens_dict = {"additional_special_tokens": gps_tokens}
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    check_token_ids(tokenizer, [gps_token_start_id, gps_token_start_id+1, gps_token_start_id+2])

    # -----------------------------
    # Load the pre-trained FusionGeoEncoder
    # -----------------------------
    fusion_geo_encoder = FusionGeoEncoder(
        gps_embed_dim=128,       # must match the dimension used in training
        quadkey_length=25,
        n=6,
        vocab_size=4096,
        fourier_output_dim=128,  # must match
        final_dim=4096           # final dimension = 4096
    )
    fusion_geo_encoder.load_state_dict(torch.load(data_args.fusion_geoencoder_path, map_location=device))
    fusion_geo_encoder.to(device)
    fusion_geo_encoder.eval()

    # Attach hook for GPS tokens
    apply_fusion_embedding_hook(model, gps_mapping, gps_ids, gps_token_start_id, fusion_geo_encoder)

    # Freeze model params
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

        # Enable additional trainable params
        trainable_param_names = training_args.trainable_params.split(",")
        for name, param in model.named_parameters():
            if any(tp in name for tp in trainable_param_names):
                param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Trainable parameter names: {[name for name, p in model.named_parameters() if p.requires_grad]}")

    # Prepare data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_module['data_collator'],
        train_dataset=data_module['train_dataset'],
    )

    print("Starting training with FusionGeoEncoder...")
    trainer.train(resume_from_checkpoint=False)

    torch.cuda.empty_cache()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    # Save trainable parameters
    trainable_state_dict = {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
    torch.save(trainable_state_dict, os.path.join(training_args.output_dir, "trainable_params.bin"))
    print(f"Trainable parameters saved to {os.path.join(training_args.output_dir, 'trainable_params.bin')}")

if __name__ == "__main__":
    train()