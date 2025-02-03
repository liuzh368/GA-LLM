# test_geo_encoder.py
import torch
import torch.nn as nn
import sys
import os

# 添加 geo_model 目录到系统路径，以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from geo_encoder import GeoEncoder

def main():
    # 指定保存的 GeoEncoder 模型的路径
    model_path = '/home/liuzhao/demo/LLM4POI/outputs/nyc/geo_encoder_11_10_16/8/geo_encoder_merged.pth'

    # 初始化 GeoEncoder
    # 请确保参数与训练时使用的参数一致
    gps_embed_dim = 4096  # 根据您的模型设置，如果不同请调整
    num_gps = 4096
    quadkey_length = 25
    n = 6

    geo_encoder = GeoEncoder(
        gps_embed_dim=gps_embed_dim,
        num_gps=num_gps,
        quadkey_length=quadkey_length,
        n=n
    )

    # 加载保存的模型状态字典
    state_dict = torch.load(model_path, map_location='cpu')
    geo_encoder.load_state_dict(state_dict)

    # 输出模型架构
    print("GeoEncoder 模型架构：")
    print(geo_encoder)

    # 设置模型为评估模式
    geo_encoder.eval()

    # 输入示例经纬度
    latitude = 40.7128    # 纽约市纬度
    longitude = -74.0060  # 纽约市经度

    # 生成嵌入
    with torch.no_grad():
        embedding = geo_encoder.generate_embedding(latitude, longitude)

    # 输出嵌入的形状和内容
    print("\n生成的嵌入形状：", embedding.shape)
    print("生成的嵌入向量：")
    print(embedding)

if __name__ == '__main__':
    main()