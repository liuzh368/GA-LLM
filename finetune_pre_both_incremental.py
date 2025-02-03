# finetune_pre_both.py

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
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
from llama_attn_replace_sft import replace_llama_attn  # Replace LLAMA attention mechanism

from transformers import TrainerCallback
import random

# Import GeoEncoder
from geo_model.geo_encoder import GeoEncoder

import pandas as pd

# Set random seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensure the same convolution algorithms are used every time
    torch.backends.cudnn.benchmark = False

# Call set_seed at the beginning of the code
set_seed(42)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
os.environ["WANDB_DISABLED"] = "true"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load data from a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")
    linear_mapping_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the pretrained linear mapping layer (optional)."}
    )
    geo_encoder_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the pretrained GeoEncoder (optional)."}
    )
    peft_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to the directory containing LoRA fine-tuned parameters (optional)."}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    poi_embedding_path: str = field(
        default=None, metadata={"help": "Path to the POI embeddings (.npy file)."}
    )
    gps_mapping_path: str = field(
        default=None, metadata={"help": "Path to the GPS mapping CSV file."}
    )

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
        metadata={"help": "Whether to use low-rank adaptation (LoRA) for training."},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of update steps to accumulate before performing a backward/update pass."},
    )
    deepspeed: str = field(
        default="ds_configs/stage2.json",
        metadata={"help": "Path to the DeepSpeed config file."}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir. Default is 3."}
    )
    # LoRA configurations
    lora_r_main_model: int = field(
        default=8,
        metadata={"help": "Rank for the LoRA low-rank matrix approximation for the main model and linear_mapping_layer."}
    )
    lora_alpha_main_model: float = field(
        default=32,
        metadata={"help": "Alpha scaling factor for combining LoRA weights for the main model and linear_mapping_layer."}
    )
    lora_r_geo_encoder: int = field(
        default=8,
        metadata={"help": "Rank for the LoRA low-rank matrix approximation for the GeoEncoder."}
    )
    lora_alpha_geo_encoder: float = field(
        default=32,
        metadata={"help": "Alpha scaling factor for combining LoRA weights for the GeoEncoder."}
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
        label[:source_len] = IGNORE_INDEX  # Mask the `question` part for loss calculation
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

def initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, device):
    """Initialize the POI embedding layer."""
    with torch.no_grad():
        poi_embedding_layer.to(device)
        poi_embeddings = poi_embeddings.to(device)

        # Store the original 128-dimensional POI embeddings
        poi_embedding_layer.weight.data = poi_embeddings
    print("Initialized 128-dimensional POI embeddings.")

def load_gps_mapping(gps_mapping_path):
    """Load the GPS mapping from a CSV file."""
    gps_mapping_df = pd.read_csv(gps_mapping_path)
    gps_mapping = gps_mapping_df.set_index('GPS_id').T.to_dict('list')
    return gps_mapping

def embedding_hook(module, inputs, output):
    """Hook function to dynamically process POI and GPS tokens."""
    input_ids = inputs[0].to(module.weight.device)
    token_embedding = output.clone()

    # Get token ID ranges
    poi_token_start_id = getattr(module, 'poi_token_start_id', None)
    gps_token_start_id = getattr(module, 'gps_token_start_id', None)
    max_token_id = getattr(module, 'max_token_id', None)

    # Process POI tokens
    if poi_token_start_id is not None and gps_token_start_id is not None:
        is_poi_token = (input_ids >= poi_token_start_id) & (input_ids < gps_token_start_id)
        poi_token_ids = input_ids[is_poi_token] - poi_token_start_id
        if poi_token_ids.size(0) > 0:
            # Map original 128-dimensional POI embeddings to high-dimensional space
            original_poi_embedding = module.poi_embedding_layer(poi_token_ids)
            poi_token_embedding = module.linear_mapping_layer(original_poi_embedding)
            token_embedding[is_poi_token] = poi_token_embedding.to(token_embedding.dtype)
    else:
        is_poi_token = torch.zeros_like(input_ids, dtype=torch.bool)

    # Process GPS tokens
    if gps_token_start_id is not None and max_token_id is not None:
        is_gps_token = (input_ids >= gps_token_start_id) & (input_ids <= max_token_id)
        is_gps_token = is_gps_token & (~is_poi_token)  # Exclude POI tokens
        gps_token_ids = input_ids[is_gps_token] - gps_token_start_id
        if gps_token_ids.size(0) > 0:
            # Get corresponding GPS ids
            gps_ids = [module.gps_id_list[token_id] for token_id in gps_token_ids.cpu().numpy()]
            # Get corresponding latitude and longitude coordinates
            gps_coordinates = [module.gps_mapping[gps_id] for gps_id in gps_ids]
            # Use GeoEncoder to generate embeddings
            geo_embeddings = []
            for lat, lng in gps_coordinates:
                embedding = module.geo_encoder.generate_embedding(lat, lng)
                geo_embeddings.append(embedding.squeeze(0))
            geo_embeddings = torch.stack(geo_embeddings).to(module.weight.device)
            # Replace embeddings at corresponding positions
            token_embedding[is_gps_token] = geo_embeddings.to(token_embedding.dtype)

    return token_embedding

def apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id, gps_mapping, gps_id_list, gps_token_start_id, geo_encoder, max_token_id):
    """Add hook function to the model's embedding layer."""

    # Adjusted to handle nested models
    def get_input_embeddings(model):
        if hasattr(model, 'get_input_embeddings'):
            return model.get_input_embeddings()
        elif hasattr(model, 'model') and hasattr(model.model, 'get_input_embeddings'):
            return model.model.get_input_embeddings()
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'get_input_embeddings'):
            return model.base_model.get_input_embeddings()
        elif hasattr(model, 'module') and hasattr(model.module, 'get_input_embeddings'):
            return model.module.get_input_embeddings()
        else:
            raise AttributeError("get_input_embeddings method not found in the model.")

    embedding_layer = get_input_embeddings(model)
    embedding_layer.max_token_id = max_token_id
    # For POI processing
    if poi_embedding_layer is not None:
        embedding_layer.poi_embedding_layer = poi_embedding_layer
        embedding_layer.poi_token_start_id = poi_token_start_id
        # Adjusted to get linear_mapping_layer
        def get_linear_mapping_layer(model):
            if hasattr(model, 'linear_mapping_layer'):
                print("Accessed linear_mapping_layer at model.linear_mapping_layer")
                return model.linear_mapping_layer
            elif hasattr(model, 'model') and hasattr(model.model, 'linear_mapping_layer'):
                print("Accessed linear_mapping_layer at model.model.linear_mapping_layer")
                return model.model.linear_mapping_layer
            elif hasattr(model, 'base_model') and hasattr(model.base_model.model, 'linear_mapping_layer'):
                print("Accessed linear_mapping_layer at model.base_model.model.linear_mapping_layer")
                return model.base_model.model.linear_mapping_layer
            else:
                # Try recursively accessing the modules
                for name, module in model.named_modules():
                    if hasattr(module, 'linear_mapping_layer'):
                        print(f"Found linear_mapping_layer at: {name}.linear_mapping_layer")
                        return module.linear_mapping_layer
                raise AttributeError("linear_mapping_layer not found in the model.")
        # Use this function to get linear_mapping_layer
        embedding_layer.linear_mapping_layer = get_linear_mapping_layer(model)
    # For GPS processing
    if geo_encoder is not None:
        embedding_layer.gps_mapping = gps_mapping
        embedding_layer.gps_id_list = gps_id_list
        embedding_layer.gps_token_start_id = gps_token_start_id
        # Adjusted to get geo_encoder
        def get_geo_encoder(model):
            if hasattr(model, 'geo_encoder'):
                print("Accessed geo_encoder at model.geo_encoder")
                return model.geo_encoder
            elif hasattr(model, 'model') and hasattr(model.model, 'geo_encoder'):
                print("Accessed geo_encoder at model.model.geo_encoder")
                return model.model.geo_encoder
            elif hasattr(model, 'base_model') and hasattr(model.base_model.model, 'geo_encoder'):
                print("Accessed geo_encoder at model.base_model.model.geo_encoder")
                return model.base_model.model.geo_encoder
            else:
                # Try recursively accessing the modules
                for name, module in model.named_modules():
                    if hasattr(module, 'geo_encoder'):
                        print(f"Found geo_encoder at: {name}.geo_encoder")
                        return module.geo_encoder
                raise AttributeError("geo_encoder not found in the model.")
        # Use this function to get geo_encoder
        embedding_layer.geo_encoder = get_geo_encoder(model)
    embedding_layer.register_forward_hook(embedding_hook)

def check_token_ids(tokenizer, ids_to_check):
    print("Checking Token ID mappings:")
    for token_id in ids_to_check:
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"Token ID {token_id}: {token}")

# Custom Trainer class
class POITrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)

        # Get output directory
        checkpoint_folder = f"{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        # Save merged linear_mapping_layer
        try:
            save_combined_linear_mapping_layer(model, output_dir)
            print(f"Merged linear_mapping_layer saved to {output_dir}")
        except AttributeError as e:
            print(f"Error saving linear_mapping_layer: {e}")

        # Save merged GeoEncoder
        try:
            geo_encoder = get_geo_encoder(model)
            # Unwrap if it's a PeftModel
            if isinstance(geo_encoder, PeftModel):
                print("Unwrapping PeftModel to get base geo_encoder for saving")
                base_geo_encoder = geo_encoder.get_base_model()
            else:
                base_geo_encoder = geo_encoder
            # Merge LoRA weights into base_geo_encoder
            merge_lora_weights_geo_encoder(base_geo_encoder)
            # Remove residual LoRA modules
            remove_lora_from_state_dict(base_geo_encoder)
            # Save the merged base_geo_encoder
            geo_encoder_save_path = os.path.join(output_dir, "geo_encoder_merged.pth")
            torch.save(base_geo_encoder.state_dict(), geo_encoder_save_path)
            print(f"Merged GeoEncoder saved to {geo_encoder_save_path}")
        except AttributeError as e:
            print(f"Error saving GeoEncoder: {e}")

        # Keep only the latest checkpoints
        checkpoints = [os.path.join(self.args.output_dir, d) for d in os.listdir(self.args.output_dir) if d.isdigit()]
        checkpoints = sorted(checkpoints, key=lambda x: int(os.path.basename(x)))
        if len(checkpoints) > self.args.save_total_limit:
            checkpoints_to_delete = checkpoints[:-self.args.save_total_limit]
            for checkpoint in checkpoints_to_delete:
                print(f"Deleting older checkpoint: {checkpoint}")
                try:
                    os.system(f"rm -rf {checkpoint}")
                except Exception as e:
                    print(f"Error deleting checkpoint {checkpoint}: {e}")

# Function to save and merge the linear_mapping_layer
def save_combined_linear_mapping_layer(model, output_path):
    # Merge LoRA weights into linear_mapping_layer
    with torch.no_grad():
        # Define function to get linear_mapping_layer
        def get_linear_mapping_layer(model):
            if hasattr(model, 'linear_mapping_layer'):
                print("Accessed linear_mapping_layer at model.linear_mapping_layer")
                return model.linear_mapping_layer
            elif hasattr(model, 'model') and hasattr(model.model, 'linear_mapping_layer'):
                print("Accessed linear_mapping_layer at model.model.linear_mapping_layer")
                return model.model.linear_mapping_layer
            elif hasattr(model, 'base_model') and hasattr(model.base_model.model, 'linear_mapping_layer'):
                print("Accessed linear_mapping_layer at model.base_model.model.linear_mapping_layer")
                return model.base_model.model.linear_mapping_layer
            else:
                # Try recursively accessing the modules
                for name, module in model.named_modules():
                    if hasattr(module, 'linear_mapping_layer'):
                        print(f"Found linear_mapping_layer at: {name}.linear_mapping_layer")
                        return module.linear_mapping_layer
                raise AttributeError("linear_mapping_layer not found in the model.")

        module = get_linear_mapping_layer(model)
        # If module is a PeftModel, get the base model
        if isinstance(module, PeftModel):
            print("Unwrapping PeftModel to get base linear_mapping_layer")
            module = module.get_base_model()
        # Merge LoRA weights into base module
        if hasattr(module, 'merge_and_unload'):
            module.merge_and_unload()
        # Remove residual LoRA modules
        remove_lora_from_state_dict(module)
        # Save merged linear_mapping_layer weights
        combined_weight_path = os.path.join(output_path, "linear_mapping_layer_combined.pt")
        torch.save(module.state_dict(), combined_weight_path)
        print(f"Combined linear_mapping_layer weights saved to: {combined_weight_path}")

# Function to merge LoRA weights in geo_encoder
def merge_lora_weights_geo_encoder(geo_encoder):
    # Merge LoRA weights into base_geo_encoder
    from peft.utils import _get_submodules
    if hasattr(geo_encoder, 'merge_and_unload'):
        geo_encoder.merge_and_unload()
    else:
        # For each module, check if it has merge_and_unload
        for name, module in geo_encoder.named_modules():
            if hasattr(module, 'merge_and_unload'):
                module.merge_and_unload()

# Function to remove residual LoRA modules from the model
def remove_lora_from_state_dict(model):
    """
    Remove residual LoRA parameters from the model's state_dict.
    """
    with torch.no_grad():
        keys_to_delete = []
        for name, param in model.state_dict().items():
            if 'lora_' in name:
                keys_to_delete.append(name)
        for key in keys_to_delete:
            del model.state_dict()[key]

def get_geo_encoder(model):
    if hasattr(model, 'geo_encoder'):
        print("Accessed geo_encoder at model.geo_encoder")
        return model.geo_encoder
    elif hasattr(model, 'model') and hasattr(model.model, 'geo_encoder'):
        print("Accessed geo_encoder at model.model.geo_encoder")
        return model.model.geo_encoder
    elif hasattr(model, 'base_model') and hasattr(model.base_model.model, 'geo_encoder'):
        print("Accessed geo_encoder at model.base_model.model.geo_encoder")
        return model.base_model.model.geo_encoder
    else:
        # Try recursively accessing the modules
        for name, module in model.named_modules():
            if hasattr(module, 'geo_encoder'):
                print(f"Found geo_encoder at: {name}.geo_encoder")
                return module.geo_encoder
        raise AttributeError("geo_encoder not found in the model.")

def train():
    # Parse command-line arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Ensure that poi_embedding_path and gps_mapping_path are provided
    if data_args.poi_embedding_path is None or data_args.gps_mapping_path is None:
        raise ValueError("Both poi_embedding_path and gps_mapping_path must be provided.")

    # Print all parameter configurations
    print("Model Arguments:")
    print(f"  Model Name or Path: {model_args.model_name_or_path}")
    print(f"  Model Type: {model_args.model_type}")
    print(f"  Linear Mapping Path: {model_args.linear_mapping_path}")
    print(f"  GeoEncoder Path: {model_args.geo_encoder_path}")
    print(f"  PEFT Directory: {model_args.peft_dir}")

    print("\nData Arguments:")
    print(f"  Data Path: {data_args.data_path}")
    print(f"  POI Embedding Path: {data_args.poi_embedding_path}")
    print(f"  GPS Mapping Path: {data_args.gps_mapping_path}")

    print("\nTraining Arguments:")
    print(f"  Cache Directory: {training_args.cache_dir}")
    print(f"  Optimizer: {training_args.optim}")
    print(f"  Model Max Length: {training_args.model_max_length}")
    print(f"  Use Flash Attention: {training_args.use_flash_attn}")
    print(f"  Use Full Attention: {training_args.use_full_attn}")
    print(f"  Low Rank Training: {training_args.low_rank_training}")
    print(f"  Gradient Accumulation Steps: {training_args.gradient_accumulation_steps}")
    print(f"  DeepSpeed Config: {training_args.deepspeed}")
    print(f"  Save Total Limit: {training_args.save_total_limit}")
    print(f"  LoRA r (Main Model and Linear Mapping Layer): {training_args.lora_r_main_model}, Alpha: {training_args.lora_alpha_main_model}")
    print(f"  LoRA r (GeoEncoder): {training_args.lora_r_geo_encoder}, Alpha: {training_args.lora_alpha_geo_encoder}")

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

    # Load model with 4-bit quantization (QLoRA)
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

    # Load trainable_params and LoRA parameters if peft_dir is provided
    if model_args.peft_dir:
        trainable_params_path = os.path.join(model_args.peft_dir, "trainable_params.bin")
        if os.path.isfile(trainable_params_path):
            model.load_state_dict(torch.load(trainable_params_path, map_location=model.device), strict=False)
            print("trainable_params loaded successfully!")

        # Load LoRA parameters (PeftModel)
        model = PeftModel.from_pretrained(model, model_args.peft_dir, device_map="auto", torch_dtype=torch.float16)
        print("Loaded LLM fine-tuned parameters (including LoRA) successfully.")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # Print the model structure after loading PeftModel
    print("\nModel structure after loading PeftModel:")
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")

    # Prepare special tokens for POI and GPS
    special_tokens_dict = {}
    max_token_id = max(tokenizer.get_vocab().values())

    # Initialize variables
    poi_token_start_id = None
    gps_token_start_id = None
    poi_embeddings = None
    poi_embedding_layer = None
    geo_encoder = None
    gps_mapping = None
    gps_ids = None

    # Load and process POI embeddings
    poi_embeddings = load_poi_embeddings(data_args.poi_embedding_path)
    num_poi_tokens = len(poi_embeddings)
    poi_token_start_id = max_token_id + 1
    max_token_id += num_poi_tokens

    # Create POI tokens
    if "nyc" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(0, num_poi_tokens)]
    elif "ca" in data_args.data_path.lower() or "tky" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(1, num_poi_tokens + 1)]
    else:
        # Default token names
        poi_tokens = [f"<POI {i}>" for i in range(num_poi_tokens)]

    special_tokens_dict["additional_special_tokens"] = special_tokens_dict.get("additional_special_tokens", []) + poi_tokens

    # Initialize POI embedding layer
    poi_embedding_layer = nn.Embedding(num_embeddings=num_poi_tokens, embedding_dim=128)  # Original 128-dimensional embeddings
    device = model.device if torch.cuda.is_available() else torch.device('cpu')
    initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, device)

    # Create or load linear_mapping_layer
    linear_mapping_layer = nn.Linear(128, model.config.hidden_size)
    if model_args.linear_mapping_path:
        # Load pretrained linear_mapping_layer
        linear_mapping_state = torch.load(model_args.linear_mapping_path, map_location=device)
        linear_mapping_layer.load_state_dict(linear_mapping_state)
        print(f"Loaded pretrained linear_mapping_layer from {model_args.linear_mapping_path}")
    linear_mapping_layer = linear_mapping_layer.to(device)

    # Attach linear_mapping_layer to the model
    def set_linear_mapping_layer(model, linear_mapping_layer):
        # Try to set linear_mapping_layer at various possible locations
        if hasattr(model, 'base_model') and hasattr(model.base_model.model, 'model'):
            model.base_model.model.model.linear_mapping_layer = linear_mapping_layer
            print("Set linear_mapping_layer in model.base_model.model.model")
        elif hasattr(model, 'base_model') and hasattr(model.base_model.model, 'linear_mapping_layer'):
            model.base_model.model.linear_mapping_layer = linear_mapping_layer
            print("Set linear_mapping_layer in model.base_model.model")
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'linear_mapping_layer'):
            model.base_model.linear_mapping_layer = linear_mapping_layer
            print("Set linear_mapping_layer in model.base_model")
        elif hasattr(model, 'model') and hasattr(model.model, 'linear_mapping_layer'):
            model.model.linear_mapping_layer = linear_mapping_layer
            print("Set linear_mapping_layer in model.model")
        elif hasattr(model, 'linear_mapping_layer'):
            model.linear_mapping_layer = linear_mapping_layer
            print("Set linear_mapping_layer in model")
        else:
            # If not found, try setting it in the embedding layer
            embedding_layer = model.get_input_embeddings()
            embedding_layer.linear_mapping_layer = linear_mapping_layer
            print("Set linear_mapping_layer in the embedding layer.")

    set_linear_mapping_layer(model, linear_mapping_layer)

    # Load and process GPS mapping
    gps_mapping = load_gps_mapping(data_args.gps_mapping_path)
    num_gps_tokens = len(gps_mapping)
    gps_token_start_id = max_token_id + 1
    max_token_id += num_gps_tokens

    # Create GPS tokens
    gps_ids = list(gps_mapping.keys())
    if "nyc" in data_args.data_path.lower():
        gps_tokens = [f"<GPS {i}>" for i in range(0, num_gps_tokens)]
    elif "ca" in data_args.data_path.lower() or "tky" in data_args.data_path.lower():
        gps_tokens = [f"<GPS {i}>" for i in range(1, num_gps_tokens + 1)]
    else:
        # Default token names
        gps_tokens = [f"<GPS {i}>" for i in range(num_gps_tokens)]

    special_tokens_dict["additional_special_tokens"] = special_tokens_dict.get("additional_special_tokens", []) + gps_tokens

    # Initialize GeoEncoder
    geo_encoder = GeoEncoder(gps_embed_dim=model.config.hidden_size)
    if model_args.geo_encoder_path:
        # Load pretrained GeoEncoder
        geo_encoder_state = torch.load(model_args.geo_encoder_path, map_location=device)
        geo_encoder.load_state_dict(geo_encoder_state)
        print(f"Loaded pretrained GeoEncoder from {model_args.geo_encoder_path}")
    geo_encoder = geo_encoder.to(device)

    # Attach geo_encoder to the model
    def set_geo_encoder(model, geo_encoder):
        # Try to set geo_encoder at various possible locations
        if hasattr(model, 'base_model') and hasattr(model.base_model.model, 'model'):
            model.base_model.model.model.geo_encoder = geo_encoder
            print("Set geo_encoder in model.base_model.model.model")
        elif hasattr(model, 'base_model') and hasattr(model.base_model.model, 'geo_encoder'):
            model.base_model.model.geo_encoder = geo_encoder
            print("Set geo_encoder in model.base_model.model")
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'geo_encoder'):
            model.base_model.geo_encoder = geo_encoder
            print("Set geo_encoder in model.base_model")
        elif hasattr(model, 'model') and hasattr(model.model, 'geo_encoder'):
            model.model.geo_encoder = geo_encoder
            print("Set geo_encoder in model.model")
        elif hasattr(model, 'geo_encoder'):
            model.geo_encoder = geo_encoder
            print("Set geo_encoder in model")
        else:
            # Try recursively accessing the modules
            for name, module in model.named_modules():
                if hasattr(module, 'geo_encoder'):
                    print(f"Found geo_encoder at: {name}.geo_encoder")
                    setattr(module, 'geo_encoder', geo_encoder)
                    return
            # If still not found, raise an error
            raise AttributeError("Cannot set geo_encoder in the model.")

    set_geo_encoder(model, geo_encoder)

    # Verify linear_mapping_layer attachment
    try:
        linear_mapping = getattr(model.base_model.model.model, 'linear_mapping_layer', None)
        if linear_mapping is not None:
            print("linear_mapping_layer attached successfully.")
        else:
            print("linear_mapping_layer not found in the model.")
    except AttributeError:
        print("linear_mapping_layer not found in the model.")

    # Verify geo_encoder attachment
    try:
        geo_enc = getattr(model.base_model.model.model, 'geo_encoder', None)
        if geo_enc is not None:
            print("geo_encoder attached successfully.")
        else:
            print("geo_encoder not found in the model.")
    except AttributeError:
        print("geo_encoder not found in the model.")

    # Adjust tokenizer and embedding size
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    # Check token IDs
    check_token_ids(tokenizer, [poi_token_start_id, gps_token_start_id])

    # Apply embedding hook
    gps_id_list = gps_ids if gps_mapping else []
    apply_embedding_hook(
        model,
        poi_embedding_layer,
        poi_token_start_id,
        gps_mapping if gps_mapping else {},
        gps_id_list,
        gps_token_start_id,
        geo_encoder,
        max_token_id
    )

    # Freeze model parameters except for LoRA and custom layers
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Enable requires_grad for linear_mapping_layer and geo_encoder
    linear_mapping_layer.requires_grad = True

    for param in geo_encoder.parameters():
        param.requires_grad = True

    # Apply LoRA to the main model, including linear_mapping_layer
    if training_args.low_rank_training:
        # Include 'linear_mapping_layer' in target_modules
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", ".*linear_mapping_layer.*"]
        lora_config_main = LoraConfig(
            r=training_args.lora_r_main_model,
            lora_alpha=training_args.lora_alpha_main_model,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config_main)

    # Ensure linear_mapping_layer is on the same device as the model
    linear_mapping_layer.to(device)

    # Print out the module names in geo_encoder
    print("\nModules in geo_encoder:")
    for name, module in geo_encoder.named_modules():
        print(f" - {name}: {module.__class__.__name__}")

    # Adjust target_modules for geo_encoder based on actual module names
    lora_config_geo = LoraConfig(
        r=training_args.lora_r_geo_encoder,
        lora_alpha=training_args.lora_alpha_geo_encoder,
        target_modules=[
            "gps_embed_model.gps_embedding",
            "gps_encoder.transformer_encoder.layers.*.self_attn.out_proj",
            "gps_encoder.transformer_encoder.layers.*.linear1",
            "gps_encoder.transformer_encoder.layers.*.linear2",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    geo_encoder = get_peft_model(geo_encoder, lora_config_geo)
    geo_encoder = geo_encoder.to(device)  # Ensure geo_encoder is on the correct device

    # Update the geo_encoder in the model
    set_geo_encoder(model, geo_encoder)

    # Verify trainable parameters
    print("\nThe following parameters will be trained:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # Create data module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Disable cache and enable gradient checkpointing
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Use POITrainer
    trainer = POITrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_module['data_collator'],
        train_dataset=data_module['train_dataset'],
    )

    print("\nTraining starts...")
    print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")

    # Start training
    trainer.train(resume_from_checkpoint=False)

    torch.cuda.empty_cache()

    trainer.save_state()

    # Save final merged linear_mapping_layer and GeoEncoder
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Merge and save linear_mapping_layer
    save_combined_linear_mapping_layer(model, output_dir)
    print(f"Final merged linear_mapping_layer saved to {output_dir}")

    # Merge and save GeoEncoder
    geo_encoder = get_geo_encoder(model)
    # Unwrap if it's a PeftModel
    if isinstance(geo_encoder, PeftModel):
        print("Unwrapping PeftModel to get base geo_encoder for saving")
        base_geo_encoder = geo_encoder.get_base_model()
    else:
        base_geo_encoder = geo_encoder
    merge_lora_weights_geo_encoder(base_geo_encoder)
    remove_lora_from_state_dict(base_geo_encoder)
    geo_encoder_save_path = os.path.join(output_dir, "geo_encoder_final_merged.pth")
    torch.save(base_geo_encoder.state_dict(), geo_encoder_save_path)
    print(f"Final merged GeoEncoder saved to {geo_encoder_save_path}")

if __name__ == "__main__":
    train()