# test_next_both_fusion.py

import os
import math
import torch
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
from peft import PeftModel
from llama_attn_replace_sft import replace_llama_attn
from transformers import BitsAndBytesConfig
import re
from geo_model.fusion_geo_encoder import FusionGeoEncoder4096  # 修改导入
from torch import nn

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def parse_config():
    parser = argparse.ArgumentParser(description='Next POI and GPS prediction test')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size during inference')
    parser.add_argument('--base_model', type=str, default="./model", help='Path to the base model')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='Path to the cache directory')
    parser.add_argument('--seq_len', type=int, default=32768, help='Context length during evaluation')
    parser.add_argument('--context_size', type=int, default=32768, help='Context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=True, help='Enable flash attention')
    parser.add_argument('--model_path', type=str, default='./model', help='Path to the model')
    parser.add_argument('--data_path', type=str, default="./dataset", help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--dataset_name', type=str, default="ca", help='Dataset name')
    parser.add_argument('--test_file', type=str, default="test_qa_pairs_kqt_200items_nyc_llm_without_sim_gps_poiwocoor.txt", help='Test file name')
    parser.add_argument('--test_type', type=str, default="llm", choices=["base", "projector", "llm"], help='Type of test')
    parser.add_argument('--poi_embedding_path', type=str, default=None, help='Path to the POI embeddings (.npy file)')
    parser.add_argument('--projector_path', type=str, default=None, help='Path to the trained projector')
    parser.add_argument('--gps_mapping_path', type=str, default=None, help='Path to the GPS mapping CSV file')
    parser.add_argument('--fusion_geoencoder_path', type=str, default=None, help='Path to the trained FusionGeoEncoder')  # 修改参数名
    parser.add_argument('--use_random_projector', type=str, default="False", help='Use a randomly initialized projector')
    args = parser.parse_args()
    return args

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and model embeddings."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def load_poi_embeddings(poi_embedding_path):
    """Load the POI embeddings from a .npy file."""
    poi_embeddings = np.load(poi_embedding_path)
    return torch.tensor(poi_embeddings)

def initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, linear_mapping_layer, device):
    """Initialize the POI embedding layer with pre-trained POI embeddings and apply the linear mapping."""
    with torch.no_grad():
        poi_embedding_layer.to(device)
        linear_mapping_layer.to(device)  # 确保 linear_mapping_layer 在设备上
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
    fusion_geo_encoder = module.fusion_geo_encoder  # 使用 fusion_geo_encoder

    # 确保 fusion_geo_encoder 在正确的设备上
    if fusion_geo_encoder is not None:
        fusion_geo_encoder_device = next(fusion_geo_encoder.parameters()).device
        assert fusion_geo_encoder_device == module.weight.device, "FusionGeoEncoder 不在正确的设备上!"
    else:
        print("Warning: fusion_geo_encoder is None!")

    # Define the token ranges for POI and GPS
    is_poi_token = (input_ids >= poi_token_start_id) & (input_ids < gps_token_start_id)
    is_gps_token = input_ids >= gps_token_start_id

    # Handle POI tokens
    poi_token_ids = input_ids[is_poi_token] - poi_token_start_id
    if poi_token_ids.size(0) > 0:
        poi_token_ids = poi_token_ids.to(poi_embedding_layer.device)
        if poi_token_ids.max().item() < poi_embedding_layer.size(0):
            poi_token_embedding = poi_embedding_layer[poi_token_ids]
            token_embedding[is_poi_token] = poi_token_embedding.to(token_embedding.dtype)
        else:
            print("Error: POI token ID exceeds embedding layer size!")

    # Handle GPS tokens
    gps_token_ids = input_ids[is_gps_token] - gps_token_start_id
    if gps_token_ids.size(0) > 0:
        gps_ids = [gps_id_list[token_id] for token_id in gps_token_ids.cpu().numpy()]
        gps_coordinates = [gps_mapping[gps_id] for gps_id in gps_ids]

        fusion_embeddings = []
        for idx, (lat, lng) in enumerate(gps_coordinates):
            print(f"Generating geo embedding for GPS token {idx}: latitude={lat}, longitude={lng}")
            try:
                # FusionGeoEncoder4096 expects individual lat and lng
                emb_4096 = fusion_geo_encoder(latitude=lat, longitude=lng)  # [1,4096]
                print(f"Generated geo embedding shape: {emb_4096.shape}")
                fusion_embeddings.append(emb_4096.squeeze(0))
            except AttributeError as e:
                print(f"Error generating geo embeddings: {e}")
            except Exception as e:
                print(f"Unexpected error generating geo embeddings: {e}")

        if fusion_embeddings:
            fusion_embeddings = torch.stack(fusion_embeddings).to(module.weight.device)
            token_embedding[is_gps_token] = fusion_embeddings.to(token_embedding.dtype)
        else:
            print("No valid fusion embeddings generated.")

    return token_embedding

def apply_embedding_hook(model, poi_embedding_layer, poi_token_start_id, gps_mapping, gps_id_list, gps_token_start_id, fusion_geo_encoder):
    """Apply embedding hook to handle POI and GPS tokens."""
    embedding_layer = model.get_input_embeddings()
    embedding_layer.poi_embedding_layer = poi_embedding_layer
    embedding_layer.poi_token_start_id = poi_token_start_id
    embedding_layer.gps_mapping = gps_mapping
    embedding_layer.gps_id_list = gps_id_list
    embedding_layer.gps_token_start_id = gps_token_start_id
    embedding_layer.fusion_geo_encoder = fusion_geo_encoder
    embedding_layer.register_forward_hook(embedding_hook)

def load_projector(projector_path, input_dim=128, output_dim=4096):
    """Load the pre-trained projector (linear mapping layer)."""
    linear_mapping_layer = nn.Linear(input_dim, output_dim)
    state_dict = torch.load(projector_path, map_location="cpu")  # Load to CPU first
    linear_mapping_layer.load_state_dict(state_dict)
    return linear_mapping_layer

def evaluate_prediction_accuracy(prediction, ground_truth):
    """Evaluate prediction accuracy."""
    pred_poi_pattern1 = r"POI id (\d+)."
    if "POI id" in prediction:
        match = re.search(pred_poi_pattern1, prediction)
        predicted_poi = match.group(1) if match else None
    elif "." in prediction:
        predicted_poi = prediction[:-1]
    else:
        predicted_poi = prediction

    match = re.search(pred_poi_pattern1, ground_truth)
    actual_poi = match.group(1) if match else None

    if predicted_poi is None or actual_poi is None:
        return 0

    print(f"Predicted POI ID: {predicted_poi}, Actual POI ID: {actual_poi}")
    return int(predicted_poi == actual_poi)

def remove_lora_keys(state_dict):
    """
    Remove keys containing 'lora_' from the state_dict.
    """
    keys_to_remove = [key for key in state_dict.keys() if 'lora_' in key]
    for key in keys_to_remove:
        del state_dict[key]
    return state_dict

def main(args):
    print("========== Configuration Parameters ==========")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("==============================================")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.seq_len,
        padding_side="right",
        use_fast=True,
    )

    if args.flash_attn:
        replace_llama_attn(inference=True)

    config = transformers.AutoConfig.from_pretrained(args.base_model)
    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
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

    if args.test_type in ["projector", "llm"]:
        # Load POI embeddings
        poi_embeddings = load_poi_embeddings(args.poi_embedding_path)
        linear_mapping_layer = load_projector(args.projector_path)
        # Move to device
        linear_mapping_layer = linear_mapping_layer.to(device)
        max_token_id = max(tokenizer.get_vocab().values())
        poi_token_start_id = max_token_id + 1
        num_poi_tokens = len(poi_embeddings)
        model_embedding_dim = model.get_input_embeddings().weight.shape[1]
        poi_embedding_layer = nn.Embedding(num_embeddings=num_poi_tokens, embedding_dim=model_embedding_dim).to(device)
        initialize_poi_embeddings(poi_embedding_layer, poi_embeddings, linear_mapping_layer, device)

        # Load GPS mapping
        gps_mapping = load_gps_mapping(args.gps_mapping_path)
        gps_ids = list(gps_mapping.keys())
        num_gps_tokens = len(gps_ids)
        gps_token_start_id = poi_token_start_id + num_poi_tokens

        # Initialize FusionGeoEncoder4096
        fusion_geo_encoder = FusionGeoEncoder4096(
            gps_embed_dim=128,          # n-gram/transformer 分支的 embedding 大小
            quadkey_length=25,         # QuadKey 长度
            n=6,                       # n-gram 的 n
            vocab_size=4096,           # n-gram embedding 中的 Embedding 大小
            fourier_output_dim=128,    # 可学习的傅里叶分支输出大小
            final_dim=config.hidden_size,  # 最终映射到与 LLM hidden_size 一致
            dropout=0.1
        ).to(device)

        # Load FusionGeoEncoder weights
        if args.fusion_geoencoder_path:
            fusion_geoencoder_state_dict = torch.load(args.fusion_geoencoder_path, map_location=device)
            # 根据实际情况确定是否需要去除前缀
            def strip_prefix_from_state_dict(state_dict, prefix):
                """
                Remove specified prefix from state_dict keys.
                """
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith(prefix):
                        new_key = key[len(prefix):]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                return new_state_dict

            # 假设需要去除 "base_model.model." 前缀
            fusion_geoencoder_state_dict = strip_prefix_from_state_dict(fusion_geoencoder_state_dict, "base_model.model.")

            # 移除包含 'lora_' 的键（如果存在）
            fusion_geoencoder_state_dict = remove_lora_keys(fusion_geoencoder_state_dict)

            # 加载 state_dict，设置 strict=False 以避免因缺少键导致的错误
            try:
                fusion_geo_encoder.load_state_dict(fusion_geoencoder_state_dict, strict=False)
                fusion_geo_encoder.to(device)
                fusion_geo_encoder.eval()
                print(f"Successfully loaded FusionGeoEncoder weights from {args.fusion_geoencoder_path}")
            except Exception as e:
                print(f"Error loading FusionGeoEncoder state_dict: {e}")

        # Define special tokens
        if args.dataset_name == "nyc":
            poi_tokens = [f"<POI {i}>" for i in range(0, num_poi_tokens)]
            gps_tokens = [f"<GPS {i}>" for i in range(num_gps_tokens)]
        elif args.dataset_name in ["ca", "tky"]:
            poi_tokens = [f"<POI {i}>" for i in range(1, num_poi_tokens + 1)]
            gps_tokens = [f"<GPS {i}>" for i in range(1, num_gps_tokens + 1)]
        else:
            raise ValueError("Unsupported dataset name. Please use 'nyc', 'ca', or 'tky'.")

        special_tokens_dict = {"additional_special_tokens": poi_tokens + gps_tokens}
        smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

        # Apply embedding hook
        apply_embedding_hook(
            model,
            poi_embedding_layer,
            poi_token_start_id,
            gps_mapping,
            gps_ids,
            gps_token_start_id,
            fusion_geo_encoder
        )

    if args.test_type == "llm":
        # Load PeftModel directly without loading trainable_params.bin
        trainable_params = os.path.join(args.output_dir, "trainable_params.bin")
        if os.path.isfile(trainable_params):
            # Load state_dict
            trainable_state_dict = torch.load(trainable_params, map_location=model.device)
            # 移除嵌入层相关的键
            keys_to_remove = [key for key in trainable_state_dict.keys() if 'embed_tokens.weight' in key]
            for key in keys_to_remove:
                del trainable_state_dict[key]
            # 加载剩余的 state_dict
            try:
                model.load_state_dict(trainable_state_dict, strict=False)
                print("trainable_params loaded successfully!")
            except Exception as e:
                print(f"Error loading trainable_params state_dict: {e}")

        # Load PeftModel
        try:
            model = PeftModel.from_pretrained(model, args.output_dir, device_map="auto", torch_dtype=torch.float16)
            print("Loaded LLM fine-tuned parameters (including LoRA) successfully.")
        except Exception as e:
            print(f"Error loading PeftModel: {e}")

    # Define default special tokens
    special_tokens_dict = {
        "pad_token": DEFAULT_PAD_TOKEN,
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN
    }
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    model.eval()

    # Define generation config
    generation_config = transformers.GenerationConfig(
        max_new_tokens=4,
        do_sample=True,
        temperature=0.6,
        top_k=40,
        top_p=0.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )

    # Load test data
    data_path = f'{args.data_path}/{args.dataset_name}/preprocessed/{args.test_file}'
    with open(data_path, "r") as file:
        lines = file.readlines()

    correct_predictions_1 = 0
    valid_lines_count = 0
    correct_list = []

    for index, line in tqdm(enumerate(lines), desc="Processing lines", total=len(lines)):
        try:
            prompt1, gt = line.split("<answer>:")
            tmp1, tmp2 = prompt1.split('Which POI id will user ')
            time = tmp1[-24:]
            user_id = tmp2.split(' visit')[0]
            prompt = prompt1.replace('<question>:', '<question>:') + '<answer>:' + f'{time} user {user_id} will visit POI id '

            if len(tokenizer.tokenize(prompt)) >= args.seq_len:
                continue

            valid_lines_count += 1
            prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(**prompt_inputs, generation_config=generation_config)
            torch.cuda.empty_cache()

            gt = gt.replace('[', '').replace(']', '')
            i = 0
            while i < 1:
                try:
                    output_tokens = outputs[:, prompt_inputs.input_ids.shape[1]:][i]
                    prediction = tokenizer.decode(output_tokens, skip_special_tokens=True)
                    filtered_prediction = re.sub(r'[^0-9]', '', prediction)
                    i += 1
                    tmp = evaluate_prediction_accuracy(filtered_prediction, gt)
                    if tmp:
                        if i == 1:
                            correct_list.append(index)
                            correct_predictions_1 += tmp
                            break
                except Exception as e:
                    print(f"Error decoding prediction on line {index}: {e}")
                    break

        except Exception as e:
            print(f"Error processing line {index}: {e}")
            continue

    print(f'valid_lines_count: {valid_lines_count}')
    if valid_lines_count > 0:
        print(f'ACC@1: {correct_predictions_1 / valid_lines_count}')
    else:
        print('No valid lines to evaluate.')
    print(f'correct_index: {correct_list}')

if __name__ == "__main__":
    args = parse_config()
    main(args)