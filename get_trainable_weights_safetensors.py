import os
import torch
import argparse
from safetensors.torch import save_file

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--checkpoint_path', type=str, default="/root/autodl-tmp/liuzhao_data/output_dir_poi_embedding_32768/checkpoint-109122")
    parser.add_argument('--trainable_params', type=str, default="embed,norm")
    args = parser.parse_args()
    return args

def main(args):
    path = args.checkpoint_path
    trainable_params = args.trainable_params.split(",")

    weights_all = torch.load(os.path.join(path, "pytorch_model.bin"))

    weights_trainable = {}
    weights_lora = {}
    for k in weights_all:
        print(f"Original key: {k}, shape: {weights_all[k].shape}")  # Debug print
        if "lora" in k:
            k_new = k.replace("default.", "") if "default." in k else k
            weights_lora[k_new] = weights_all[k]
        else:
            if any([n in k for n in trainable_params]):
                print(f"Selected key: {k}, shape: {weights_all[k].shape}")  # Debug print
                weights_trainable[k[17:]] = weights_all[k]

    adapter_model_safetensors = os.path.join(path, "adapter_model.safetensors")
    trainable_params_safetensors = os.path.join(path, "trainable_params.safetensors")

    # 保存为safetensors格式
    save_file(weights_lora, adapter_model_safetensors)
    save_file(weights_trainable, trainable_params_safetensors)

    # 保存为bin格式
    adapter_model_bin = os.path.join(path, "adapter_model.bin")
    trainable_params_bin = os.path.join(path, "trainable_params.bin")

    torch.save(weights_lora, adapter_model_bin)
    torch.save(weights_trainable, trainable_params_bin)

if __name__ == "__main__":
    args = parse_config()
    main(args)
