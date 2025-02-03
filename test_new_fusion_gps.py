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
import pandas as pd

# Import the FusionGeoEncoder (already placed in geo_model/fusion_geo_encoder.py)
from geo_model.fusion_geo_encoder import FusionGeoEncoder4096 as FusionGeoEncoder

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def parse_config():
    parser = argparse.ArgumentParser(description='Next POI prediction test with FusionGeoEncoder')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size during inference')
    parser.add_argument('--base_model', type=str, default="./model", help='Path to the base model')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='Path to the cache directory')
    parser.add_argument('--seq_len', type=int, default=32768, help='Context length during evaluation')
    parser.add_argument('--context_size', type=int, default=32768, help='Context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=True, help='Enable flash attention')
    parser.add_argument('--model_path', type=str, default='./model', help='Path to the model')
    parser.add_argument('--data_path', type=str, default="./dataset", help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--dataset_name', type=str, default="nyc", help='Dataset name')
    parser.add_argument('--test_file', type=str, default="test_qa_pairs.txt", help='Test file name')
    parser.add_argument('--test_type', type=str, default="llm", choices=["base", "projector", "llm"], help='Type of test')
    parser.add_argument('--gps_mapping_path', type=str, default=None, help='Path to the GPS mapping CSV file')
    parser.add_argument('--fusion_geoencoder_path', type=str, default=None, help='Path to the trained FusionGeoEncoder')
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

def fusion_embedding_hook(module, inputs, output):
    """
    Embedding hook for Next POI testing with FusionGeoEncoder.
    Replaces GPS token embeddings with the output from a pre-trained FusionGeoEncoder.
    """
    input_ids = inputs[0].to(module.weight.device)
    gps_token_start_id = module.gps_token_start_id
    gps_mapping = module.gps_mapping
    gps_id_list = module.gps_id_list
    fusion_geo_encoder = module.fusion_geo_encoder

    is_gps_token = input_ids >= gps_token_start_id
    token_embedding = output.clone()

    gps_token_ids = input_ids[is_gps_token] - gps_token_start_id
    if gps_token_ids.size(0) > 0:
        gps_ids = [gps_id_list[token_id] for token_id in gps_token_ids.cpu().numpy()]
        gps_coordinates = [gps_mapping[gps_id] for gps_id in gps_ids]

        fusion_embeddings = []
        for lat, lng in gps_coordinates:
            # Use the FusionGeoEncoder to generate embeddings
            emb_4096 = fusion_geo_encoder(latitude=lat, longitude=lng)  # [1,4096]
            fusion_embeddings.append(emb_4096.squeeze(0))

        fusion_embeddings = torch.stack(fusion_embeddings).to(module.weight.device)
        token_embedding[is_gps_token] = fusion_embeddings.to(token_embedding.dtype)

    return token_embedding

def apply_fusion_embedding_hook(model, gps_mapping, gps_id_list, gps_token_start_id, fusion_geo_encoder):
    """Attach the fusion embedding hook to the model's embedding layer."""
    embedding_layer = model.get_input_embeddings()
    embedding_layer.gps_mapping = gps_mapping
    embedding_layer.gps_id_list = gps_id_list
    embedding_layer.gps_token_start_id = gps_token_start_id
    embedding_layer.fusion_geo_encoder = fusion_geo_encoder
    embedding_layer.register_forward_hook(fusion_embedding_hook)

def evaluate_prediction_accuracy(prediction, ground_truth):
    pred_poi_pattern = r"POI id (\d+)."
    if "POI id" in prediction:
        match_pred = re.search(pred_poi_pattern, prediction)
        predicted_poi = match_pred.group(1) if match_pred else prediction
    elif "." in prediction:
        predicted_poi = prediction[:-1]
    else:
        predicted_poi = prediction

    match_gt = re.search(pred_poi_pattern, ground_truth)
    actual_poi = match_gt.group(1) if match_gt else ground_truth

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

    # Replace attention if flash_attn is needed
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
        gps_token_start_id = max(tokenizer.get_vocab().values()) + 1
        gps_id_list = list(gps_mapping.keys())

        # Create GPS tokens
        if args.dataset_name == "nyc":
            gps_tokens = [f"<GPS {i}>" for i in range(len(gps_mapping))]
        elif args.dataset_name in ["ca", "tky"]:
            gps_tokens = [f"<GPS {i}>" for i in range(1, len(gps_mapping) + 1)]
        else:
            raise ValueError("Unsupported dataset name. Use 'nyc', 'ca', or 'tky'.")

        tokenizer.add_special_tokens({"additional_special_tokens": gps_tokens})
        model.resize_token_embeddings(len(tokenizer))

        if args.use_random_projector.lower() == 'true':
            print("Using a randomly initialized FusionGeoEncoder (just for demonstration).")
            fusion_geo_encoder = FusionGeoEncoder()  # random weights
        else:
            print("Loading pretrained FusionGeoEncoder from path.")
            fusion_geo_encoder = FusionGeoEncoder(
                gps_embed_dim=128,    # Must match training config
                quadkey_length=25,
                n=6,
                vocab_size=4096,
                fourier_output_dim=128,
                final_dim=4096
            )
            fusion_geo_encoder.load_state_dict(torch.load(args.fusion_geoencoder_path, map_location=device))
            fusion_geo_encoder.to(device)
            fusion_geo_encoder.eval()

        # Attach hook
        apply_fusion_embedding_hook(model, gps_mapping, gps_id_list, gps_token_start_id, fusion_geo_encoder)

    if args.test_type == "llm":
        # Load LoRA (trainable_params)
        trainable_params_file = os.path.join(args.output_dir, "trainable_params.bin")
        if os.path.isfile(trainable_params_file):
            model.load_state_dict(torch.load(trainable_params_file, map_location=model.device), strict=False)
            print("trainable_params loaded successfully!")

        model = PeftModel.from_pretrained(model, args.output_dir, device_map="auto", torch_dtype=torch.float16)
        print("Loaded LLM fine-tuned parameters (including LoRA) successfully.")

    # Add special tokens
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