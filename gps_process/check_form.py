import numpy as np

# 加载保存的 .npy 文件
npy_path = '/home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_embeddings.npy'
gps_embeddings = np.load(npy_path, allow_pickle=True)  # 将其加载为字典格式

# 显示POI数量和部分embedding内容
print("总的POI数量:", len(gps_embeddings))

# 显示前几个POI的GPS embedding，检查形状和数值
print("部分POI的 GPS embedding 内容:")
for i, (poi_id, gps_embedding) in enumerate(gps_embeddings.items()):
    print(f"POI ID: {poi_id}")
    print(f"GPS embedding shape: {gps_embedding.shape}")
    print(f"GPS embedding (部分内容): {gps_embedding[:5]}")  # 打印 embedding 的前5个元素作为示例
    if i >= 4:  # 显示前5个POI的信息
        break

# 检查所有GPS embedding是否一致
embedding_shapes = {gps_embedding.shape for gps_embedding in gps_embeddings.values()}
print("所有GPS embedding的形状一致:", len(embedding_shapes) == 1, "形状:", embedding_shapes)