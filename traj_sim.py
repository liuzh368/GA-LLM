import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import random
import argparse
import sys
import pickle as pkl
import heapq

import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from llama_attn_replace_sft import replace_llama_attn
# from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


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
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="./model")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=32768, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=32768, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=True, help='')
    parser.add_argument('--data_path', type=str, default="./datasets", help='')
    parser.add_argument('--output_dir', type=str, default="/home/liuz/demo/LLM4POI/output_dir",
                        help='')
    parser.add_argument('--dataset_name', type=str, default="ca",
                        help='')
    args = parser.parse_args()
    return args


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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


def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx + batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i + seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y


def iceildiv(x, y):
    return (x + y - 1) // y


def compute_features(hidden, attention):
    averaged_attention = attention.mean(dim=1)
    weighted_hidden_states = torch.zeros_like(hidden)
    batch_size, sequence_length, hidden_size = hidden.shape
    for i in range(batch_size):
        # For each example, perform a weighted sum of hidden states
        # based on the attention weights
        for j in range(sequence_length):
            weighted_hidden_states[i, j, :] = torch.matmul(
                averaged_attention[i, j, :],
                hidden[i, :, :]
            )
    weighted_hidden_states = weighted_hidden_states.mean(axis=[0, 1])
    return weighted_hidden_states


def main(args):
    device = "cuda:0"
    seed = 2
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_path = './model'
    output_dir = args.output_dir
    print("data path", args.data_path)
    print("base model", model_path)
    print("peft model", output_dir)

    # tokenizer = transformers.AutoTokenizer.from_pretrained(...)：从 model_path 加载预训练的分词器（Tokenizer）。
    # 设置 model_max_length 为 32768，表示模型输入的最大长度。padding_side="right" 表示在右侧进行填充，
    # use_fast=True 启用快速分词器。local_files_only=True 表示只使用本地文件，不从网络下载。
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=32768,
        padding_side="right",
        use_fast=True,
        local_files_only=True  # 添加这个参数，明确表示使用本地文件
    )

    # print(tokenizer('6', return_tensors="pt").to(device))
    # print(tokenizer.decode([    29946]))
    # sys.exit()
    # if args.flash_attn:
    #     replace_llama_attn(inference=True)

    # Set RoPE scaling factor
    # config = transformers.AutoConfig.from_pretrained(...)：从 model_path 加载模型配置文件。
    # output_hidden_states=True 和 output_attentions=True 表示模型在前向传播时输出隐藏状态和注意力权重，
    # _flash_attn_2_enabled=True 启用 flash attention 相关配置。
    config = transformers.AutoConfig.from_pretrained(
        model_path,
        cache_dir=None,
        output_hidden_states=True,
        output_attentions=True,
        _flash_attn_2_enabled=True
    )
    # context_size = args.context_size if args.context_size > 0 else args.seq_len：设置上下文大小。
    # 如果传入的 context_size 大于 0，则使用它；否则使用 seq_len。
    context_size = args.context_size if args.context_size > 0 else args.seq_len
    # orig_ctx_len = getattr(config, "max_position_embeddings", None)：获取模型的最大位置编码（max_position_embeddings）长度，
    # 对于 LLaMA2 模型来说，通常是 4096。
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
    # 如果 context_size 大于 orig_ctx_len，则计算缩放因子，并在配置中启用相应的缩放。
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # model = transformers.AutoModelForCausalLM.from_pretrained(...)：从 model_path 加载一个用于因果语言建模的预训练模型。
    # device_map='auto' 自动分配模型的设备，torch_dtype=torch.bfloat16 指定模型权重的精度，使用 bfloat16 以节省内存。
    # BitsAndBytesConfig 配置量化策略，使用 4 位精度。
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        config=config,
        cache_dir=None,
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
    # 调整模型的词嵌入矩阵大小为 32001。
    model.resize_token_embeddings(32001)

    # special_tokens_dict：初始化一个字典，用于存放特殊符号（pad_token、eos_token、bos_token、unk_token）。
    special_tokens_dict = dict()
    # 如果 tokenizer 没有定义这些特殊符号，则将其设置为默认值。
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    # smart_tokenizer_and_embedding_resize(...)：调整 tokenizer 和 model 的词嵌入矩阵大小，以适应新的特殊符号。
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # 指定 LoRA 需要调整的模型参数，这些通常是注意力机制的投影矩阵。
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # config = LoraConfig(...)：配置 LoRA 的参数，r=8 是 LoRA 的秩，lora_alpha=16 是缩放因子，
    # target_modules=targets 指定要应用 LoRA 的模块，task_type="CAUSAL_LM" 指定任务类型为因果语言模型（Causal Language Model）。
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=targets,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # 将模型与 LoRA 配置结合，生成一个可训练的模型。
    model = get_peft_model(model, config)
    # 将模型切换到评估模式（evaluation mode），关闭 dropout 层等在训练时启用的操作。
    model.eval()

    # data_path = f'datasets/preprocessed/{args.dataset_name}/'
    data_path = f'datasets/{args.dataset_name}/preprocessed/'

    # 这是一个用于计算特征的函数，它根据 train 参数来处理训练或测试数据。
    def compute_fea(train=True):
        # key_query_traj = {}：初始化一个空字典，用于存储 key 和 query 的特征向量。
        key_query_traj = {}
        # output = 'train' if train else 'test'：根据 train 参数决定处理的是训练数据还是测试数据。
        if train:
            output = 'train'
        else:
            output = 'test'
        if train:
            list_data_dict = jload(data_path + f'{output}_kq_pairs.json')
        else:
            list_data_dict = jload(data_path + f'{output}_kq_pairs.json')

        # 遍历数据列表 list_data_dict，并显示进度条。
        for e in tqdm(list_data_dict, desc="Processing lines", total=len(list_data_dict)):
            try:
                # 将 e['key'] 文本通过 tokenizer 转换为张量，并移动到指定的 GPU 设备上。
                key = tokenizer(e['key'], return_tensors="pt").to(device)
                # key = model(**key)：通过模型前向传播计算出 key 的隐藏状态和注意力权重。
                key = model(**key)
                # 计算 key 的特征向量，并移到 CPU 上。
                key = compute_features(key.hidden_states[-1], key.attentions[-1]).cpu().detach()
                # 清空 CUDA 的缓存，释放显存。
                torch.cuda.empty_cache()
                # 对 query 进行同样的处理，但在第二块 GPU 上进行。
                query = tokenizer(e['query'], return_tensors="pt").to('cuda:1')
                query = model(**query)
                query = compute_features(query.hidden_states[-1], query.attentions[-1]).cpu().detach()
                torch.cuda.empty_cache()
                # key_query_traj[e['traj_id']] = {'key': key, 'query': query, 'start_time': e['start_time'], 'end_time': e['end_time']}：
                # 将计算出的 key 和 query 特征向量以及其他信息存入 key_query_traj 字典中。
                key_query_traj[e['traj_id']] = {'key': key, 'query': query, 'start_time': e['start_time'], 'end_time':e['end_time']}
            except Exception as ex:
                print(f"An error occurred: {ex}")  # Log the exception
                continue
        # 将字典 key_query_traj 保存为 pickle 文件。
        with open(data_path + f'{output}_kqt.pkl', 'wb') as fp:
            pkl.dump(key_query_traj, fp)

    compute_fea(True)
    compute_fea(False)
    # 计算相似度的函数。
    def compute_sim(train=True):
        if train:
            with open(data_path + 'train_kqt.pkl', 'rb') as fp:
                key_query_traj_train = pkl.load(fp)
            with open(data_path + 'train_kqt.pkl', 'rb') as fp:
                key_query_traj = pkl.load(fp)
        else:
            with open(data_path + 'train_kqt.pkl', 'rb') as fp:
                key_query_traj_train = pkl.load(fp)
            with open(data_path + 'test_kqt.pkl', 'rb') as fp:
                key_query_traj = pkl.load(fp)
        # Assuming key_query_traj is already populated with PyTorch tensors
        # 初始化一个空字典，用于存储计算结果。
        results = {}
        # 获取可用的 GPU 设备。
        gpus = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

        # Function to compute similarity for a subset of data on a specific GPU
        # 这个函数接受一个数据子集，并在指定的 GPU 上计算 key 与 query 的相似度，并返回本地结果。
        def compute_similarity_on_gpu(subset, other_subset, gpu):
            # # 初始化一个字典，用于存储每个 key 与最相似的 query 轨迹 ID
            local_results = {}

            # Stack all query tensors for batch processing
            # 将所有 query 的张量堆叠在一起，以便批量处理
            # 将所有的 query 从 CPU 移动到指定的 GPU 上，并调整形状为 (1, -1)，即一个平坦化的向量
            query_tensors = [other_element['query'].to(gpu).reshape(1, -1) for other_element in other_subset.values()]
            # 将所有 query 的张量垂直堆叠为一个大张量
            stacked_queries = torch.vstack(query_tensors)

            # 这段代码生成了一个布尔矩阵 time_condition_matrix，表示每个 key 是否满足与对应 query 的时间条件。
            # 将 start_times 和 end_times 从字符串转换为浮点数，并在 GPU 上创建张量
            start_times = torch.tensor([float(element['start_time']) for element in subset.values()], device=gpu)
            end_times = torch.tensor([float(other_element['end_time']) for other_element in other_subset.values()],
                                     device=gpu)

            # Pre-compute a matrix for time condition checks
            # 这个矩阵表示每个 key 的 start_time 是否大于对应的 query 的 end_time，用于时间过滤
            time_condition_matrix = start_times[:, None] > end_times

            # 遍历 subset 中的每个 key，计算与其他 query 的相似度
            for traj_id, element in tqdm(subset.items(), desc="Computing Similarities"):
                key_tensor = element['key'].to(gpu).reshape(1, -1)

                # 根据时间条件过滤掉不符合条件的 query
                # valid_indices 是一个布尔向量，表示哪些 query 符合时间条件
                valid_indices = time_condition_matrix[list(subset.keys()).index(traj_id)]
                # 根据 valid_indices 过滤 stacked_queries，得到有效的 query
                filtered_queries = stacked_queries[valid_indices]
                # 如果没有符合条件的 query，则跳过这个 key
                if len(filtered_queries) == 0:
                    continue

                # # 计算 key 与过滤后的 query 之间的相似度，这里使用余弦相似度
                batch_similarities = F.cosine_similarity(key_tensor, filtered_queries)

                # 提取前 35 个最相似的 query 的相似度值和索引
                top_k = min(len(filtered_queries), 35)

                # 如果有有效的相似度计算结果，则提取 top_k 个最相似的 query
                if top_k > 0:
                    # 获取 top_k 个最大相似度的索引
                    _, top_indices = torch.topk(batch_similarities, k=top_k)
                    # 根据 top_indices 获取对应的 query 轨迹 ID，并存储在 local_results 中
                    top_queries_traj_ids = [list(other_subset.keys())[idx] for idx in top_indices.cpu().numpy()]
                    # 将 traj_id 对应的 top_queries_traj_ids 存储在 local_results 中
                    local_results[traj_id] = top_queries_traj_ids
                else:
                    # 如果没有有效的相似度，则设置为空列表
                    local_results[traj_id] = []
            # 返回包含 key 与最相似 query 对应轨迹 ID 的字典
            return local_results
        # Divide the data among GPUs
        # 为每个 GPU 分配一个数据子集。
        data_subsets = {gpu: {} for gpu in gpus}
        # 将 key_query_traj 中的每个轨迹分配给一个 GPU。
        for i, (traj_id, data) in enumerate(key_query_traj.items()):
            gpu = gpus[i % len(gpus)]
            data_subsets[gpu][traj_id] = data

        # Initialize a global progress bar
        total_tasks = len(key_query_traj)
        progress_bar = tqdm(total=total_tasks, desc="Overall Progress")

        # Compute similarities in parallel on multiple GPUs
        # 并行地在多个 GPU 上计算相似度。
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = []
            for gpu in gpus:
                # 将计算任务提交到线程池中。
                future = executor.submit(compute_similarity_on_gpu, data_subsets[gpu], key_query_traj_train, gpu)
                futures.append(future)
                # Update the progress bar immediately after task submission
                progress_bar.update(1)

            # Retrieve results from completed futures
            for future in futures:
                local_results = future.result()
                # 将每个线程的局部结果合并到全局结果中。
                results.update(local_results)

        progress_bar.close()
        if train:
            with open(data_path + 'train_key_top200.json', 'w') as fp:
                json.dump(results, fp)
        else:
            with open(data_path + 'test_key_top200.json', 'w') as fp:
                json.dump(results, fp)
    compute_sim(True)
    compute_sim(False)

if __name__ == "__main__":
    args = parse_config()
    main(args)