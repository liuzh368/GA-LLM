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
# from transformers import unwrap_model

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

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
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

# Define POIModel with additional embeddings
class POIModel(transformers.LlamaForCausalLM):
    def __init__(self, config, poi_embeddings=None, poi_token_start_id=None):
        super(POIModel, self).__init__(config)
        # Only set the layers if poi_embeddings and poi_token_start_id are provided
        if poi_embeddings is not None and poi_token_start_id is not None:
            self.poi_embedding_layer = nn.Embedding.from_pretrained(poi_embeddings, freeze=True)
            self.linear_mapping_layer = nn.Linear(128, config.hidden_size)
            self.poi_token_start_id = poi_token_start_id

    @classmethod
    def from_pretrained_with_poi(cls, pretrained_model_name_or_path, poi_embeddings, poi_token_start_id, **kwargs):
        # Load the model without poi_embeddings and poi_token_start_id to avoid __init__ errors
        model = super(POIModel, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        # Manually set the custom layers after loading the model
        model.poi_embedding_layer = nn.Embedding.from_pretrained(poi_embeddings, freeze=True)
        model.linear_mapping_layer = nn.Linear(128, model.config.hidden_size)
        model.poi_token_start_id = poi_token_start_id
        return model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        is_poi_token = input_ids >= self.poi_token_start_id
        poi_token_ids = input_ids[is_poi_token] - self.poi_token_start_id

        if poi_token_ids.size(0) > 0:
            poi_embeddings = self.poi_embedding_layer(poi_token_ids)
            poi_embeddings.requires_grad = True  # 确保 poi_embeddings 允许梯度计算
            mapped_poi_embeddings = self.linear_mapping_layer(poi_embeddings)

            # 保留 mapped_poi_embeddings 的梯度
            mapped_poi_embeddings.retain_grad()

            # 将 mapped_poi_embeddings 赋值到 inputs_embeds 的指定位置
            inputs_embeds[is_poi_token] = mapped_poi_embeddings

            # 打印 inputs_embeds 的部分值，确认修改
            print("inputs_embeds sample after modification:", inputs_embeds[is_poi_token][:5])

        # 确保 kwargs 中没有 inputs_embeds，避免冲突
        kwargs.pop("inputs_embeds", None)

        # 调用父类 forward 函数并返回结果
        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

class POITrainer(Trainer):

    @staticmethod
    def unwrap_model(model):
        """Recursively unwraps a model from potential containers (e.g., DDP, DeepSpeed)."""
        if hasattr(model, "module"):
            return POITrainer.unwrap_model(model.module)
        else:
            return model

    def create_optimizer_and_scheduler(self, num_training_steps):
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad]}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        self.lr_scheduler = self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

        # 检查是否有有效的参数
        if not any([len(group["params"]) > 0 for group in optimizer_grouped_parameters]):
            raise ValueError("Optimizer has no parameters with gradients.")

        return self.optimizer, self.lr_scheduler

    def training_step(self, model, inputs):
        # 执行前向传播并计算损失
        outputs = model(**inputs)
        loss = outputs.loss



        loss.backward()  # 反向传播计算梯度

        # 检查 linear_mapping_layer 的权重梯度是否被有效传递
        unwrapped_model = self.unwrap_model(model)
        if hasattr(unwrapped_model, "linear_mapping_layer"):
            # 直接对 linear_mapping_layer.weight 计算梯度，检查其依赖关系
            test_grad = torch.autograd.grad(
                outputs=loss,
                inputs=unwrapped_model.linear_mapping_layer.weight,
                retain_graph=True,
                allow_unused=True  # 若梯度为 None 则说明未有效引入损失计算
            )
            print("Test grad for linear_mapping_layer weight:",
                  test_grad[0][:5].cpu().numpy() if test_grad[0] is not None else "No gradient computed")

        return loss


def train():
    # 解析命令行参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # 加载模型的配置文件
    config = transformers.LlamaConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    poi_embeddings = load_poi_embeddings(data_args.poi_embedding_path)
    assert poi_embeddings.shape[1] == 128, "POI embeddings must be 128-dimensional."

    if "nyc" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(0, poi_embeddings.size(0))]
    elif "ca" in data_args.data_path.lower() or "tky" in data_args.data_path.lower():
        poi_tokens = [f"<POI {i}>" for i in range(1, poi_embeddings.size(0) + 1)]
    else:
        raise ValueError("Unsupported dataset name. Please use 'nyc', 'ca', or 'tky'.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    special_tokens_dict = {"additional_special_tokens": poi_tokens}

    max_token_id = max(tokenizer.get_vocab().values())
    poi_token_start_id = max_token_id + 1

    # 使用预训练模型加载
    model = POIModel.from_pretrained_with_poi(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        poi_embeddings=poi_embeddings,
        poi_token_start_id=poi_token_start_id,
        config=config,
        cache_dir=training_args.cache_dir
    ).to(torch.bfloat16)

    # 调整tokenizer和embedding大小
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.linear_mapping_layer.parameters():
        param.requires_grad = True

    # 检查 linear_mapping_layer 是否设置为可训练
    print("linear_mapping_layer requires_grad:", model.linear_mapping_layer.weight.requires_grad)

    if training_args.low_rank_training:
        lora_config = LoraConfig(
            r=1,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = POITrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_module['data_collator'],
        train_dataset=data_module['train_dataset']
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    torch.save(model.linear_mapping_layer.state_dict(), os.path.join(training_args.output_dir, "linear_mapping_layer.pt"))

if __name__ == "__main__":
    train()