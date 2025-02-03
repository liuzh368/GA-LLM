# Written by Peibo Li
# Original code based on https://github.com/dvlab-research/LongLoRA
# Licensed under the Apache License, Version 2.0

import os
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from peft import PeftModel
# from llama_attn_replace import replace_llama_attn
from llama_attn_replace_sft import replace_llama_attn
from typing import Dict, Optional, Sequence
import sys
from transformers import BitsAndBytesConfig
import re

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="/root/autodl-tmp/models/model")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=32768, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=32768, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=True, help='')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/models/model', help='your model path')
    parser.add_argument('--data_path', type=str, default="/root/autodl-tmp/liuzhao_data/POI_dataset", help='')
    parser.add_argument('--output_dir', type=str, default="/root/autodl-tmp/liuzhao_data/output_dir_poi_embedding_32768_mtnetv1/checkpoint-109122", help='')
    parser.add_argument('--dataset_name', type=str, default="ca", help='')
    parser.add_argument('--test_file', type=str, default="cleaned_test_qa_pairs_kqt_output_geo.txt", help='')
    args = parser.parse_args()
    return args

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

def evaluate(model, data, batch_size, device, seq_length, sliding_window=256, use_cache=False):
    stats = {}

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []

    with torch.no_grad():
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)
        for idx, (x, y) in tqdm(
                enumerate(
                    get_as_batch(
                        data['val'],
                        seq_length,
                        batch_size,
                        device=device,
                        sliding_window=sliding_window
                    )
                ),
                total=iceildiv(
                    iceildiv(len(data['val']), sliding_window),
                    batch_size
                )
        ):
            val_loss = 0.
            acc = 0.
            cnt = 0

            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                part_len = x[:, i:i + seq_length].shape[1]
                outputs = model(
                    input_ids=x[:, i:i + seq_length],
                    labels=x[:, i:i + seq_length].contiguous(),
                    use_cache=use_cache)

                val_loss = outputs.loss * part_len + val_loss
                acc = ((outputs.logits.argmax(-1) == y[:, i:i + seq_length]).float().sum()) + acc
                cnt += part_len
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(outputs.loss.item())
            val_loss /= cnt
            acc /= cnt

            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1))

    return stats

def main(args):

    # 打印所有传入的配置参数
    print("Configuration parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    device = "cuda:0"
    seed = 2
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_path = args.model_path
    output_dir = args.output_dir


    # 加载微调后的 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(output_dir)
    print(f"Tokenizer max length: {tokenizer.model_max_length}")

    if args.flash_attn:
        replace_llama_attn(inference=True)

    config = transformers.AutoConfig.from_pretrained(model_path)
    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # 加载基础模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    # 调整模型的 token 嵌入层大小以匹配 tokenizer 的词汇表大小
    model.resize_token_embeddings(len(tokenizer))

    # 如果存在微调模型权重，则加载它们
    trainable_params = os.path.join(output_dir, "trainable_params.bin")
    if os.path.isfile(trainable_params):
        model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)

    # 加载微调后的参数 (PEFT)
    model = PeftModel.from_pretrained(
        model,
        output_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.eval()

    # 定义用于评估预测准确率的函数
    def evaluate_prediction_accuracy(prediction, ground_truth):
        # Regular expression to extract POI ids from prediction and ground truth
        pred_poi_pattern1 = r"POI id (\d+)."  # 使用正则表达式提取 POI id
        pred_poi_pattern2 = r"(\d+)."
        # pred_poi_pattern = r"with POI id (\d+)."
        # pred_poi_pattern = r"visited POI id (\d+) with Category Name"
        # pred_poi_pattern = r'will visit POI ([^\.]+)\.'
        # pred_poi_pattern = r'will visit POI ([^\.]+) which is'
        # Extract predicted and actual POI ids
        if "POI id" in prediction:  # 提取预测和实际的 POI id
            predicted_poi = re.search(pred_poi_pattern1, prediction).group(1)
        elif "." in prediction:
            predicted_poi = prediction[:-1]
        else:
            predicted_poi = prediction
        actual_poi = re.search(pred_poi_pattern1, ground_truth).group(1)
        # predicted_poi = prediction[:-1]

        # Compare and return accuracy (1 if they match, 0 otherwise) # 比较并返回准确率
        print(f"Predicted POI ID: {predicted_poi}, Actual POI ID: {actual_poi}")

        return int(predicted_poi == actual_poi)

    # 设置生成配置
    generation_config = transformers.GenerationConfig(
        max_new_tokens=4,
        min_new_tokens=False,
        do_sample=True,
        temperature=0.6,
        top_k=40,
        top_p=0.1,
        typical_p=1.0,
        repetition_penalty=1.176,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False
    )

    data_path = f'datasets/{args.dataset_name}/preprocessed/'
    # data_path = f'/root/autodl-tmp/liuzhao_data/POI_dataset/ca/preprocessed/'
    with open(data_path + f"{args.test_file}", "r") as file:
        lines = file.readlines()

    correct_predictions_1 = 0
    model.eval()
    device = 'cuda'

    # 遍历每一行并生成预测
    correct_list = []
    valid_lines_count = 0  # 用来记录实际处理的行数
    for index, line in tqdm(enumerate(lines), desc="Processing lines", total=len(lines)):
        prompt1, gt = line.split("<answer>:")

        # 恢复时间和用户ID的提取逻辑
        tmp1, tmp2 = prompt1.split('Which POI id will user ')
        time = tmp1[-24:]
        user_id = tmp2.split(' visit')[0]

        # 构建预测问题的prompt
        prompt = prompt1.replace('<question>:',
                                 '<question>:') + '<answer>:' + f'{time} user {user_id} will visit POI id '

        # 检查token长度，确保不会超出模型的最大长度
        if len(tokenizer.tokenize(prompt)) >= 32768:  # 32768
            continue

        # 这行没有被跳过，所以计入valid_lines_count
        valid_lines_count += 1

        # 将prompt转换为模型的输入格式
        prompt = tokenizer(prompt, return_tensors="pt").to(device)

        # 生成预测输出
        outputs = model.generate(**prompt, generation_config=generation_config)
        torch.cuda.empty_cache()  # 手动清空缓存

        # 清理ground truth中的方括号
        gt = gt.replace('[', '').replace(']', '')

        # 多次尝试生成
        i = 0
        while i < 1:
            try:
                # Step 1: 截取模型生成的输出部分
                output_tokens = outputs[:, prompt.input_ids.shape[1]:][i]

                # Step 2: 解码token序列为文本字符串
                prediction = tokenizer.decode(output_tokens, skip_special_tokens=True)

                # Step 3: 使用正则表达式过滤非数字字符
                filtered_prediction = re.sub(r'[^0-9]', '', prediction)

                i += 1

                # 比较预测和ground truth的准确性
                tmp = evaluate_prediction_accuracy(filtered_prediction, gt)

                if tmp:
                    # 记录正确的预测索引
                    if i == 1:
                        correct_list.append(index)
                        correct_predictions_1 += tmp
                        break  # 直接跳出循环
                    elif i < 6:
                        break  # 5次生成内正确则跳出循环

            except Exception as e:
                print(f"Error in prediction at line {index}: {e}")
                continue

    # 输出ACC@1的结果
    print(f'valid_lines_count:{valid_lines_count}')
    print(f'ACC@1:{correct_predictions_1 / valid_lines_count}')

    print(f'correct_index:{correct_list}')

if __name__ == "__main__":
    args = parse_config()
    main(args)
