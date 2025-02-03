# Written by Peibo Li
# Original code based on https://github.com/dvlab-research/LongLoRA?tab=readme-ov-file
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

IGNORE_INDEX = -100  # 定义了一些常量，包括忽略的索引值（-100）和几个默认的特殊标记（填充标记、结束标记、开始标记和未知标记）。
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="./model_llama3.1_storm")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=32768, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=32768, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=True, help='')
    parser.add_argument('--model_path', type=str, default='./model_llama3.1_storm', help='your model path')
    parser.add_argument('--data_path', type=str, default="./dataset", help='')
    parser.add_argument('--output_dir', type=str, default="./output_dir_llama3storm_4096/checkpoint-15591", help='')
    parser.add_argument('--dataset_name', type=str, default="ca", help='')
    parser.add_argument('--test_file', type=str, default="cleaned_test_qa_pairs_kqt.txt",
                        help='')
    args = parser.parse_args()
    return args

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):  # 该函数用于对tokenizer（分词器）和模型的嵌入层进行调整，使其能够处理新添加的特殊标记（如[PAD]、[UNK]等）。
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)  # 添加特殊标记
    model.resize_token_embeddings(len(tokenizer))  # 调整模型的嵌入层大小以适应新的标记

    if num_new_tokens > 0:  # 如果有新增的标记，将更新模型的嵌入层，并用现有嵌入的平均值填充新标记的嵌入。
        input_embeddings = model.get_input_embeddings().weight.data  # 获取输入嵌入
        output_embeddings = model.get_output_embeddings().weight.data  # 获取输出嵌入

        # 取现有 embedding 的平均值来初始化新增的 token embedding
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg  # 填充新增 token 的嵌入
        output_embeddings[-num_new_tokens:] = output_embeddings_avg  # 同时更新输出嵌入


def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):  # 该函数用于将输入数据按指定长度（seq_length）和滑动窗口生成批次数据。
    all_ix = list(range(0, len(data) - seq_length, sliding_window))  # 生成滑动窗口范围的索引列表
    all_ix.pop()  # 移除最后一个索引

    for idx in range(0, len(all_ix), batch_size):  # 使用滑动窗口技术生成批次数据，确保批次间具有一定的重叠，以便后续模型训练或评估。
        ix = all_ix[idx:idx + batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])  # 确保每批数据长度足够
        x = torch.stack([torch.from_numpy((data[i:i + seq_length]).astype(np.int64)) for i in ix])  # 输入序列
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + seq_length]).astype(np.int64)) for i in ix])  # 输出序列
        if device != 'cpu': # 如果设备不是 CPU，则将数据迁移到相应的设备上
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y  # 生成器函数，逐批返回数据


def iceildiv(x, y):  # 简单的整数除法函数，用于向上取整。
    return (x + y - 1) // y


def evaluate(model, data, batch_size, device, seq_length, sliding_window=256, use_cache=False):  # 评估模型性能的函数，计算验证集上的准确率（accuracy）、损失（loss）和困惑度（perplexity）。
    stats = {}  # 保存评估结果的字典

    model.eval()  # 切换到评估模式

    loss_list_val, acc_list = [], []  # 保存损失和准确率的列表
    loss_step_list_val = []

    with torch.no_grad():  # 禁用梯度计算以节省内存
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)  # 设置打印选项，避免科学计数法
        for idx, (x, y) in tqdm(  # 使用进度条显示处理进度
                enumerate(
                    get_as_batch(
                        data['val'],
                        seq_length,
                        batch_size,
                        device=device,
                        sliding_window=sliding_window
                    )  # 使用get_as_batch函数生成批量数据，并循环计算模型的预测结果、损失和准确率。
                ),
                total=iceildiv(
                    iceildiv(len(data['val']), sliding_window),
                    batch_size
                )
        ):
            val_loss = 0.  # 初始化验证集损失
            acc = 0.  # 初始化准确率
            cnt = 0  # 计数器

            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):  # 处理每个子序列
                part_len = x[:, i:i + seq_length].shape[1]  # 当前子序列长度
                # 使用模型计算输出
                outputs = model(
                    input_ids=x[:, i:i + seq_length],
                    labels=x[:, i:i + seq_length].contiguous(),
                    use_cache=use_cache)
                # 使用模型计算输出
                val_loss = outputs.loss * part_len + val_loss
                acc = ((outputs.logits.argmax(-1) == y[:, i:i + seq_length]).float().sum()) + acc
                cnt += part_len
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])  # 保留每步的损失
                loss_step_list_val[part_idx].append(outputs.loss.item())  # 保存当前步的损失
            val_loss /= cnt  # 计算平均损失
            acc /= cnt  # 计算平均准确率

            loss_list_val.append(val_loss.item())  # 保存验证损失
            acc_list.append(acc.item())  # 保存验证准确率
    # 统计验证集的准确率、损失和困惑度
    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']  # 困惑度
    stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1))

    return stats  # 返回评估统计信息


def main(args):
    device = "cuda:0"
    seed = 2
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 输出模型路径、数据路径等信息
    model_path = args.model_path
    output_dir = args.output_dir
    print("data path", args.data_path)
    print("base model", model_path)
    print("peft model", output_dir)

    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=32768,
        padding_side="right",
        use_fast=True,
    )

    # print(tokenizer('6', return_tensors="pt").to(device))
    # print(tokenizer.decode([    29946]))
    # sys.exit()
    if args.flash_attn:
        replace_llama_attn(inference=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_path,
        # _flash_attn_2_enabled = True,
    )
    # 根据上下文大小调整 RoPE 缩放
    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # 加载模型并启用量化（4 位）以节省显存
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
    # 调整模型的 token 嵌入层大小
    # model.resize_token_embeddings(32001)
    model.resize_token_embeddings(128257)
    model.eval()
    # 如果存在微调模型权重，则加载它们
    if output_dir:
        trainable_params = os.path.join(output_dir, "trainable_params.bin") # 这里有人指出有问题了
        if os.path.isfile(trainable_params):
            model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
        model = PeftModel.from_pretrained(
            model,
            output_dir,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    # 设置特殊 token
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # 调整 tokenizer 和模型的嵌入
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    # 设置生成配置
    generation_config = transformers.GenerationConfig(
        max_new_tokens=4,
        min_new_tokens=False,
        # Generation strategy
        do_sample=True,
        # num_beams=5,
        # num_beam_groups=5,
        # penalty_alpha=None,
        use_cache=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,

        # Hyperparameters for logit manipulation
        temperature=0.6,
        top_k=40,
        top_p=0.1,
        typical_p=1.0,
        # diversity_penalty=4.0,
        repetition_penalty=1.176,
        # length_penalty=1.0,
        # no_repeat_ngram_size=0,

        num_return_sequences=1
    )
    import re
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

    data_path = f'datasets/{args.dataset_name}/preprocessed/'  # 读取测试数据
    with open(data_path + f"{args.test_file}", "r") as file:
        lines = file.readlines()
    correct_predictions_1 = 0
    correct_predictions_5 = 0
    correct_predictions_10 = 0
    model.eval()
    device = 'cuda'
    # Iterate over each line and ask the LLM
    # 遍历每一行并生成预测
    correct_list = []
    valid_lines_count = 0  # 用来记录实际处理的行数
    for index, line in tqdm(enumerate(lines), desc="Processing lines", total=len(lines)):
        prompt1, gt = line.split("<answer>:")
        tmp1, tmp2 = prompt1.split('Which POI id will user ')
        time = tmp1[-24:]
        user_id = tmp2.split(' visit')[0]
        prompt = prompt1.replace('<question>:', '<question>:') + '<answer>:' + f'{time} user {user_id} will visit POI id '
        # prompt1, prompt2, gt = line.split("<answer>:")
        # prompt = prompt1.replace('<question>:', '<user>:\n') + "\n<assistant>:\n"
        if len(tokenizer.tokenize(prompt)) >= 4096:  # 32768
            continue

        # 这行没有被跳过，所以计入valid_lines_count
        valid_lines_count += 1

        prompt = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(**prompt, generation_config=generation_config)
        torch.cuda.empty_cache()  # 手动清空缓存，与实验结果无关
        # prediction = tokenizer.decode(outputs[:, prompt.input_ids.shape[1]:][0], skip_special_tokens=True).replace('[',
        #                                                                                                            '').replace(
        #     ']', '')
        gt = gt.replace('[', '').replace(']', '')
        i = 0
        while i < 1:
            try:
                # Step 1: 截取模型生成的输出部分
                output_tokens = outputs[:, prompt.input_ids.shape[1]:][i]

                # print(f"Selected token sequence for prediction: {output_tokens}")

                # Step 2: 解码token序列为文本字符串
                prediction = tokenizer.decode(output_tokens, skip_special_tokens=True)

                # 打印解码后的文本
                # print(f"Decoded prediction: {prediction}")

                # Step 4: 使用正则表达式过滤非数字字符（这一步可能会导致只剩下数字，需谨慎）
                filtered_prediction = re.sub(r'[^0-9]', '', prediction)

                # 打印过滤后的结果
                # print(f"Filtered prediction (only digits): {filtered_prediction}")

                # prediction = tokenizer.decode(outputs[:, prompt.input_ids.shape[1]:][i],
                #                               skip_special_tokens=True)
                # prediction = re.sub(r'[^0-9]', '', prediction)
                i += 1
                # print(prediction)
                # print(gt)
                tmp = evaluate_prediction_accuracy(filtered_prediction, gt)
                if tmp:
                    if i == 1:
                        correct_list.append(index)
                        correct_predictions_1 += tmp
                        # correct_predictions_5 += tmp
                        # correct_predictions_10 += tmp
                        break
                    elif i < 6:
                        # correct_predictions_5 += tmp
                        # correct_predictions_10 += tmp
                        break
                    # else:
                    #     correct_predictions_10 += tmp
                    #     break
            except:
                continue
        # sys.exit()


    # 输出ACC@1的结果
    print(f'valid_lines_count:{valid_lines_count}')
    print(f'ACC@1:{correct_predictions_1 / valid_lines_count}')

    # print(f'ACC@5:{correct_predictions_5 / len(lines)}')
    # print(f'ACC@10:{correct_predictions_10 / len(lines)}')
    print(f'correct_index:{correct_list}')


if __name__ == "__main__":
    args = parse_config()
    main(args)

