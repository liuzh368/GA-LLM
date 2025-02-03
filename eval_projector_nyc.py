import os
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from llama_attn_replace_sft import replace_llama_attn
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
    parser.add_argument('--flash_attn', type=bool, default=True, help='Use flash attention')
    parser.add_argument('--model_path', type=str, default='/share/home/liuzh368/demo/LLM4POI/model_llama2-7b-longlora',
                        help='your model path')
    parser.add_argument('--poi_embedding_path', type=str,
                        default="/share/home/liuzh368/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy",
                        help='Path to POI embeddings.')
    parser.add_argument('--projector_path', type=str,
                        default="/share/home/liuzh368/demo/LLM4POI/outputs/nyc/projector/linear_mapping_layer.pt",
                        help='Path to the pre-trained projector (linear layer).')
    parser.add_argument('--data_path', type=str, default="/share/home/liuzh368/demo/LLM4POI/datasets", help='Dataset path')
    parser.add_argument('--output_dir', type=str,
                        default="/share/home/liuzh368/demo/LLM4POI/outputs/output_dir_v6_poi_user_32768/checkpoint-15591",
                        help='Output directory')
    parser.add_argument('--dataset_name', type=str, default="ca", help='Dataset name')
    parser.add_argument('--test_file', type=str, default="test_qa_pairs_kqt_new2_user_poi.txt", help='Test file name')
    args = parser.parse_args()
    return args

def load_poi_embeddings(poi_embedding_path):
    """Load the POI embeddings from a .npy file."""
    poi_embeddings = np.load(poi_embedding_path)
    return torch.tensor(poi_embeddings)

def load_projector(projector_path, input_dim=128, output_dim=4096):
    """Load the pre-trained projector (linear mapping layer)."""
    linear_mapping_layer = torch.nn.Linear(input_dim, output_dim)
    state_dict = torch.load(projector_path)
    linear_mapping_layer.load_state_dict(state_dict)
    return linear_mapping_layer

def initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, linear_mapping_layer):
    """Initialize the POI embedding layer with pre-trained POI embeddings and apply the linear mapping."""
    with torch.no_grad():
        device = poi_embedding_layer.weight.device  # 获取 poi_embedding_layer 所在的设备
        # 将 POI embeddings 移动到同一个设备上
        poi_embeddings = poi_embeddings.to(device)
        # 将128维的POI embedding映射到4096维
        poi_embeddings = linear_mapping_layer(poi_embeddings).to(device)  # 将线性映射后的 embeddings 也移动到同一设备
        poi_embedding_layer.weight.data = poi_embeddings
    print("POI embeddings successfully initialized.")

def embedding_hook(module, inputs, output):
    """钩子函数，用于处理 POI token 的 embedding 处理"""
    input_ids = inputs[0].to(module.weight.device)  # 确保 input_ids 和 embedding layer 在相同的设备上
    poi_embedding_layer = module.poi_embedding_layer.weight
    poi_token_start_id = module.poi_token_start_id

    is_poi_token = input_ids >= poi_token_start_id
    token_embedding = output.clone()

    poi_token_ids = input_ids[is_poi_token] - poi_token_start_id
    if poi_token_ids.size(0) > 0:
        poi_token_ids = poi_token_ids.to(poi_embedding_layer.device)  # 移动到相同设备
        poi_token_embedding = poi_embedding_layer[poi_token_ids]
        token_embedding[is_poi_token] = poi_token_embedding.to(token_embedding.device, token_embedding.dtype)

    return token_embedding

def apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id):
    """给模型的 embedding 层添加钩子函数"""
    embedding_layer = model.get_input_embeddings()
    embedding_layer.poi_embedding_layer = poi_embedding_layer
    embedding_layer.poi_token_start_id = poi_token_start_id
    embedding_layer.register_forward_hook(embedding_hook)

# 定义用于评估预测准确率的函数
def evaluate_prediction_accuracy(prediction, ground_truth):
    """Compare predicted and ground truth POI ids"""
    pred_poi_pattern1 = r"POI id (\d+)."  # 使用正则表达式提取 POI id
    pred_poi_pattern2 = r"(\d+)."

    if "POI id" in prediction:  # 提取预测和实际的 POI id
        predicted_poi = re.search(pred_poi_pattern1, prediction).group(1)
    elif "." in prediction:
        predicted_poi = prediction[:-1]
    else:
        predicted_poi = prediction
    actual_poi = re.search(pred_poi_pattern1, ground_truth).group(1)

    # Compare and return accuracy (1 if they match, 0 otherwise) # 比较并返回准确率
    print(f"Predicted POI ID: {predicted_poi}, Actual POI ID: {actual_poi}")

    return int(predicted_poi == actual_poi)

def main(args):
    # 打印所有传入的配置参数
    print("Configuration parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    device = "cuda:0"
    torch.cuda.set_device(device)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.output_dir)
    print(f"Tokenizer max length: {tokenizer.model_max_length}")

    if args.flash_attn:
        replace_llama_attn(inference=True)

    config = transformers.AutoConfig.from_pretrained(args.model_path)
    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load base model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
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

    # Load POI embeddings and projector
    poi_embeddings = load_poi_embeddings(args.poi_embedding_path)
    linear_mapping_layer = load_projector(args.projector_path)

    # Create POI embedding layer
    model_embedding_dim = model.get_input_embeddings().weight.shape[1]
    poi_embedding_layer = torch.nn.Embedding(num_embeddings=len(poi_embeddings), embedding_dim=model_embedding_dim)

    # Initialize POI embedding layer
    initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, linear_mapping_layer)

    # Add POI tokens to tokenizer
    poi_tokens = [f"<POI {i}>" for i in range(1, len(poi_embeddings) + 1)]
    special_tokens_dict = {"additional_special_tokens": poi_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Apply embedding hook
    apply_embedding_hook(model, poi_embedding_layer, len(tokenizer) - len(poi_tokens))

    # 省略微调模型权重和trainable_params部分
    model.eval()

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

    # 加载测试数据
    data_path = f'{args.data_path}/{args.dataset_name}/preprocessed/'
    with open(data_path + f"{args.test_file}", "r") as file:
        lines = file.readlines()

    correct_predictions_1 = 0
    valid_lines_count = 0
    correct_list = []

    for index, line in tqdm(enumerate(lines), desc="Processing lines", total=len(lines)):
        prompt1, gt = line.split("<answer>:")

        # 提取时间和用户 ID
        tmp1, tmp2 = prompt1.split('Which POI id will user ')
        time = tmp1[-24:]
        user_id = tmp2.split(' visit')[0]

        # 构建问题
        prompt = prompt1.replace('<question>:',
                                 '<question>:') + '<answer>:' + f'{time} user {user_id} will visit POI id '

        # 检查 token 长度
        if len(tokenizer.tokenize(prompt)) >= 32768:
            continue

        valid_lines_count += 1
        prompt = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(**prompt, generation_config=generation_config)
        torch.cuda.empty_cache()

        gt = gt.replace('[', '').replace(']', '')

        i = 0
        while i < 1:
            try:
                output_tokens = outputs[:, prompt.input_ids.shape[1]:][i]
                prediction = tokenizer.decode(output_tokens, skip_special_tokens=True)
                filtered_prediction = re.sub(r'[^0-9]', '', prediction)

                i += 1
                tmp = evaluate_prediction_accuracy(filtered_prediction, gt)

                if tmp:
                    correct_list.append(index)
                    correct_predictions_1 += tmp
                    break

            except Exception as e:
                print(f"Error in prediction at line {index}: {e}")
                continue

    # 输出ACC@1的结果
    print(f'valid_lines_count: {valid_lines_count}')
    print(f'ACC@1: {correct_predictions_1 / valid_lines_count}')
    print(f'correct_index: {correct_list}')

if __name__ == "__main__":
    args = parse_config()
    main(args)