import torch
import numpy as np
import os.path as osp

# 设置模型文件路径
model_path = "/home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/STHGCN_nyc.pt"

# 加载 checkpoint 文件
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# 检查 checkpoint 是否是一个字典
if isinstance(checkpoint, dict):
    # 查看 'model_state_dict' 是否在 checkpoint 中
    if 'model_state_dict' in checkpoint:
        # 提取 POI embedding (前 4981 个)
        poi_embedding_weight = checkpoint['model_state_dict']['checkin_embedding_layer.poi_embedding.weight'][:4980]

        # 将 POI embedding 转换为 numpy 数组并保存为 .npy 文件
        poi_embedding_np = poi_embedding_weight.detach().cpu().numpy()
        np.save(osp.join('/home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed', 'nyc_POI_embeddings_sthgcn_128.npy'),
                poi_embedding_np)

        print("POI embedding 保存成功，文件名为 'nyc_POI_embeddings_sthgcn_128.npy'")
    else:
        print("'model_state_dict' 不在 checkpoint 中，无法提取 POI embedding。")
else:
    print("checkpoint 不是字典格式，无法读取。")