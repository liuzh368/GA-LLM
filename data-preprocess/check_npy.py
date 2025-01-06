import numpy as np
import os

# 设置 .npy 文件路径
npy_file_path = "/home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_kg_128.npy"

# 检查文件是否存在
if not os.path.exists(npy_file_path):
    print(f"文件 {npy_file_path} 不存在，请检查路径是否正确。")
else:
    # 加载 .npy 文件
    data = np.load(npy_file_path)

    # 打印文件的基本信息
    print("文件加载成功！")
    print("数据类型:", data.dtype)
    print("数组形状:", data.shape)

    # 可选：查看前几个嵌入向量的数据
    print("前5个嵌入向量的值:\n", data[:5])