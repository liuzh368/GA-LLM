import os
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from peft import PeftModel
from llama_attn_replace_sft import replace_llama_attn
from transformers import BitsAndBytesConfig
import re
from geo_model.geo_encoder import GeoEncoder
import pandas as pd

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def parse_config():
    parser = argparse.ArgumentParser(description='Next POI prediction test')
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
    parser.add_argument('--test_file', type=str, default="test_qa_pairs.txt", help='Test file name')
    parser.add_argument('--test_type', type=str, default="llm", choices=["base", "projector", "llm"], help='Type of test')
    parser.add_argument('--gps_mapping_path', type=str, default=None, help='Path to the GPS mapping CSV file')
    parser.add_argument('--geoencoder_path', type=str, default=None, help='Path to the trained GeoEncoder')
    parser.add_argument('--use_random_projector', type=str, default="False", help='Use a randomly initialized projector')
    args = parser.parse_args()
    return args

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def load_gps_mapping(gps_mapping_path):
    gps_mapping_df = pd.read_csv(gps_mapping_path)
    gps_mapping = gps_mapping_df.set_index('GPS_id').T.to_dict('list')
    return gps_mapping

def embedding_hook(module, inputs, output):
    input_ids = inputs[0].to(module.weight.device)
    gps_token_start_id = module.gps_token_start_id
    gps_mapping = module.gps_mapping
    gps_id_list = module.gps_id_list
    geo_encoder = module.geo_encoder

    is_gps_token = input_ids >= gps_token_start_id
    token_embedding = output.clone()

    gps_token_ids = input_ids[is_gps_token] - gps_token_start_id
    if gps_token_ids.size(0) > 0:
        gps_ids = [gps_id_list[token_id] for token_id in gps_token_ids.cpu().numpy()]
        gps_coordinates = [gps_mapping[gps_id] for gps_id in gps_ids]

        geo_embeddings = []
        for lat, lng in gps_coordinates:
            embedding = geo_encoder.generate_embedding(lat, lng)
            geo_embeddings.append(embedding.squeeze(0))

        geo_embeddings = torch.stack(geo_embeddings).to(module.weight.device)
        token_embedding[is_gps_token] = geo_embeddings.to(token_embedding.dtype)

    return token_embedding

def apply_embedding_hook(model, gps_mapping, gps_id_list, gps_token_start_id, geo_encoder):
    embedding_layer = model.get_input_embeddings()
    embedding_layer.gps_mapping = gps_mapping
    embedding_layer.gps_id_list = gps_id_list
    embedding_layer.gps_token_start_id = gps_token_start_id
    embedding_layer.geo_encoder = geo_encoder
    embedding_layer.register_forward_hook(embedding_hook)

def evaluate_prediction_accuracy(prediction, ground_truth):
    pred_poi_pattern1 = r"POI id (\d+)."
    if "POI id" in prediction:
        predicted_poi = re.search(pred_poi_pattern1, prediction).group(1)
    elif "." in prediction:
        predicted_poi = prediction[:-1]
    else:
        predicted_poi = prediction
    actual_poi = re.search(pred_poi_pattern1, ground_truth).group(1)
    print(f"Predicted POI ID: {predicted_poi}, Actual POI ID: {actual_poi}")
    return int(predicted_poi == actual_poi)

def main(args):
    print("========== Configuration Parameters ==========")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("==============================================")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 2
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

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
        gps_mapping = load_gps_mapping(args.gps_mapping_path)

        gps_embed_dim = 4096
        num_gps = 4096
        quadkey_length = 25
        n = 6

        if args.use_random_projector.lower() == 'true':
            print("Using a randomly initialized GeoEncoder.")
            geo_encoder = GeoEncoder(gps_embed_dim=gps_embed_dim, num_gps=num_gps, quadkey_length=quadkey_length, n=n)
            geo_encoder.apply(geo_encoder.init_weights)
        else:
            print("Loading pretrained GeoEncoder from path.")
            geo_encoder = GeoEncoder(gps_embed_dim=gps_embed_dim, num_gps=num_gps, quadkey_length=quadkey_length, n=n)
            geo_encoder.load_state_dict(torch.load(args.geoencoder_path, map_location=device))
        geo_encoder.to(device)  # 将 GeoEncoder 移动到 CUDA 设备上
        geo_encoder.eval()

        gps_token_start_id = max(tokenizer.get_vocab().values()) + 1
        gps_id_list = list(gps_mapping.keys())

        # 根据 dataset_name 生成 GPS tokens
        if args.dataset_name == "nyc":
            gps_tokens = [f"<GPS {i}>" for i in range(len(gps_mapping))]
        elif args.dataset_name in ["ca", "tky"]:
            gps_tokens = [f"<GPS {i}>" for i in range(1, len(gps_mapping) + 1)]
        else:
            raise ValueError("Unsupported dataset name. Please use 'nyc', 'ca', or 'tky'.")

        special_tokens_dict = {"additional_special_tokens": gps_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        apply_embedding_hook(model, gps_mapping, gps_id_list, gps_token_start_id, geo_encoder)

    if args.test_type == "llm":
        trainable_params = os.path.join(args.output_dir, "trainable_params.bin")
        if os.path.isfile(trainable_params):
            model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
            print("trainable_params loaded successfully!")

        model = PeftModel.from_pretrained(model, args.output_dir, device_map="auto", torch_dtype=torch.float16)
        print("Loaded LLM fine-tuned parameters (including LoRA) successfully.")

    special_tokens_dict = {
        "pad_token": DEFAULT_PAD_TOKEN,
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN
    }
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    model.eval()

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

    data_path = f'{args.data_path}/{args.dataset_name}/preprocessed/{args.test_file}'
    with open(data_path, "r") as file:
        lines = file.readlines()

    correct_predictions_1 = 0
    valid_lines_count = 0
    correct_list = []

    for index, line in tqdm(enumerate(lines), desc="Processing lines", total=len(lines)):
        prompt1, gt = line.split("<answer>:")
        tmp1, tmp2 = prompt1.split('Which POI id will user ')
        time = tmp1[-24:]
        user_id = tmp2.split(' visit')[0]
        prompt = prompt1.replace('<question>:', '<question>:') + '<answer>:' + f'{time} user {user_id} will visit POI id '

        if len(tokenizer.tokenize(prompt)) >= args.seq_len:
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
                    if i == 1:
                        correct_list.append(index)
                        correct_predictions_1 += tmp
                        break
            except:
                continue

    print(f'valid_lines_count: {valid_lines_count}')
    print(f'ACC@1: {correct_predictions_1 / valid_lines_count}')
    print(f'correct_index: {correct_list}')

if __name__ == "__main__":
    args = parse_config()
    main(args)