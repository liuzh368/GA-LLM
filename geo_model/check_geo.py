import torch
from geo_encoder import GeoEncoder


def strip_prefix_from_state_dict(state_dict, prefix):
    """
    去除 state_dict 中的指定前缀。

    Args:
        state_dict (dict): 模型的 state_dict。
        prefix (str): 需要移除的前缀。

    Returns:
        dict: 处理后的 state_dict。
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # 去掉前缀
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_and_inspect_model(gps_encoder_path, device='cpu'):
    """
    加载 GeoEncoder 模型的权重，并打印模型的具体结构和参数信息。

    Args:
        gps_encoder_path (str): 已保存的模型权重文件路径 (.pth)。
        device (str): 设备 ('cpu' 或 'cuda')。
    """
    print("初始化 GeoEncoder 模型...")

    # 设置模型参数，确保与训练时一致
    gps_embed_dim = 4096
    num_gps = 4096
    quadkey_length = 25
    n = 6

    # 初始化模型
    model = GeoEncoder(
        gps_embed_dim=gps_embed_dim,
        num_gps=num_gps,
        quadkey_length=quadkey_length,
        n=n
    )

    # 将模型转移到设备
    model.to(device)

    print("加载模型权重...")
    try:
        # 加载 state_dict
        state_dict = torch.load(gps_encoder_path, map_location=device)

        # 移除多余的前缀
        stripped_state_dict = strip_prefix_from_state_dict(state_dict, "base_model.model.")

        # 加载到模型中
        model.load_state_dict(stripped_state_dict)
        print("模型权重加载成功！")
    except Exception as e:
        print(f"模型权重加载失败，错误: {e}")
        return

    # 打印模型结构
    print("\n模型结构：")
    print(model)

    # 打印模型参数
    print("\n模型参数：")
    for name, param in model.named_parameters():
        print(f"层名称: {name}, 参数大小: {param.size()}, 设备: {param.device}")


# 主函数
if __name__ == "__main__":
    gps_encoder_path = "/home/liuzhao/demo/LLM4POI/datasets/nyc/geoencoder/geo_encoder_merged_old.pth"  # 替换为实际模型权重文件路径
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    load_and_inspect_model(gps_encoder_path=gps_encoder_path, device=device)