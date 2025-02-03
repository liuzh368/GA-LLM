# Written by Peibo Li
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

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
os.environ["WANDB_DISABLED"] = "true"

# 设置设备为GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
}


def load_poi_embeddings(poi_embedding_path):
    """加载 POI embedding 文件 (.npy 格式)"""
    poi_embeddings = np.load(poi_embedding_path)
    poi_embeddings = torch.tensor(poi_embeddings)  # 不需要调用 .to(device) 对于 4bit 量化模型
    return poi_embeddings


def replace_poi_embeddings_in_preprocessing(input_ids, tokenizer, poi_embeddings, poi_mapping_layer, model):
    """在预处理阶段将 input_ids 中代表 POI 的部分替换为推荐系统生成的 embedding"""
    input_embeddings = []

    # 获取 POI token 的 id 列表
    poi_token_ids = [tokenizer.convert_tokens_to_ids(f"<POI {i}>") for i in range(1, len(poi_embeddings) + 1)]
    print(f"POI token ids: {poi_token_ids}")  # 打印POI token id以确认是否生成正确

    for batch_idx, input_seq in enumerate(input_ids):
        input_seq_tensor = torch.tensor(input_seq, dtype=torch.long)  # 不调用 .to(device)

        # 打印 input_seq 以查看是否有POI token
        print(f"Processing batch {batch_idx}, input_seq: {input_seq}")

        # 获取该序列的 embeddings
        token_embeddings = model.get_input_embeddings()(input_seq_tensor.unsqueeze(0))

        # 仅在找到 POI token 时打印替换信息
        for token_idx, token_id in enumerate(input_seq):
            # print(f"Token index: {token_idx}, Token ID: {token_id}")  # 打印每个token ID

            if token_id.item() in poi_token_ids:
                poi_index = poi_token_ids.index(token_id.item())
                print(f"Replacing token embedding for POI at batch {batch_idx}, token index {token_idx}")
                token_embeddings[0, token_idx, :] = poi_mapping_layer(poi_embeddings[poi_index])

        input_embeddings.append(token_embeddings.squeeze(0))

    return input_embeddings



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    poi_embedding_path: str = field(default=None, metadata={"help": "Path to the POI embeddings."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=8192 * 4)
    use_flash_attn: bool = field(default=True)
    use_full_attn: bool = field(default=False)
    low_rank_training: bool = field(default=True)
    trainable_params: str = field(default="embed,norm,poi_mapping_layer")


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
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
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(input_ids=input_ids, input_ids_lens=input_ids_lens)


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, poi_embeddings: torch.Tensor, poi_mapping_layer: nn.Linear, model: transformers.PreTrainedModel) -> Dict:
    """Preprocess the data by tokenizing and replacing POI embeddings."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    input_ids = examples_tokenized["input_ids"]

    token_embeddings = replace_poi_embeddings_in_preprocessing(input_ids, tokenizer, poi_embeddings, poi_mapping_layer, model)

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_embeddings=token_embeddings, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with POI embedding replacement."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, poi_embeddings: torch.Tensor, poi_mapping_layer: nn.Linear, model: transformers.PreTrainedModel):
        super(SupervisedDataset, self).__init__()
        list_data_dict = jload(data_path)

        sources = ['<question>:' + example["question"] for example in list_data_dict]
        targets = ['<answer>:' + f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]

        data_dict = preprocess(sources, targets, tokenizer, poi_embeddings, poi_mapping_layer, model)

        self.input_embeddings = data_dict["input_embeddings"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_embeddings)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_embeddings=self.input_embeddings[i], labels=self.labels[i])


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, poi_embeddings: torch.Tensor, poi_mapping_layer: nn.Linear, model: transformers.PreTrainedModel) -> Dict:
    """Make dataset and collator for supervised fine-tuning with POI embedding replacement."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poi_embeddings=poi_embeddings, poi_mapping_layer=poi_mapping_layer, model=model)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # 加载 POI embedding 和 POI mapping layer
    poi_embeddings = load_poi_embeddings(data_args.poi_embedding_path)
    poi_mapping_layer = nn.Linear(poi_embeddings.shape[1], config.hidden_size)  # 不使用 .to(device)

    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right", use_fast=True)

    # 添加 special tokens
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # 添加自定义的 POI tokens
    poi_tokens = [f"<POI {i}>" for i in range(1, len(poi_embeddings) + 1)]
    special_tokens_dict["additional_special_tokens"] = poi_tokens

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

    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    # === 添加的代码: 打印 Tokenizer 中的特殊 token 信息 ===
    print("Additional special tokens:", tokenizer.additional_special_tokens)
    for i in range(1, 5):  # 这里只是检查部分 POI tokens，可根据需要调整
        token = f"<POI {i}>"
        print(f"Token: {token}, ID: {tokenizer.convert_tokens_to_ids(token)}")
    # === 打印结束 ===


    # 制作数据集模块，提前处理 POI embedding 替换
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, poi_embeddings=poi_embeddings, poi_mapping_layer=poi_mapping_layer, model=model)

    if training_args.low_rank_training:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train(resume_from_checkpoint=False)

    # 保存模型和 poi_mapping_layer
    torch.save(poi_mapping_layer.state_dict(), os.path.join(training_args.output_dir, "poi_mapping_layer.bin"))
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
