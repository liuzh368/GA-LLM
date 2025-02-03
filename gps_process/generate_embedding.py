import pandas as pd
import numpy as np
import torch
from geo_encoder import GeoEncoder  # 假设 GeoEncoder 类在 geo_encoder.py 中

# 初始化GeoEncoder
geo_encoder = GeoEncoder()

# 加载签到数据
data_path = '/home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_sample.csv'
df = pd.read_csv(data_path)

# 提取唯一的 POI ID 和其对应的坐标 (Latitude, Longitude)
poi_coordinates = df[['PoiId', 'Latitude', 'Longitude']].drop_duplicates(subset=['PoiId'])

# 初始化列表来存储每个POI的embedding
all_embeddings = []

# 为每个 POI 生成 embedding
for _, row in poi_coordinates.iterrows():
    poi_id = row['PoiId']
    latitude = row['Latitude']
    longitude = row['Longitude']

    # 生成 POI 的 embedding
    embedding = geo_encoder.generate_embedding(latitude, longitude)

    # 将 embedding 作为 numpy 数组加入列表
    all_embeddings.append(embedding.numpy())

# 将 embedding 列表转换为 ndarray 格式
all_embeddings_array = np.vstack(all_embeddings)  # 转换为 (4980, 128) 格式的 ndarray

# 检查最终数组形状是否符合要求
print("Embedding array shape:", all_embeddings_array.shape)

# 将结果保存到 .npy 文件
output_path = '/home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_embeddings.npy'
np.save(output_path, all_embeddings_array)

print(f"POI embeddings 已保存到 {output_path}")