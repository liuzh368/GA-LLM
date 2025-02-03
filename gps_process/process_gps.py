import torch
import numpy as np
from gps_model import GPSEncoder, GPSEmbeddings
from quad_key_encoder import latlng2quadkey
from itertools import product
from nltk import ngrams

# 初始化参数
gps_embed_dim = 128  # 嵌入维度
num_gps = 4096  # 假设有4096个GPS位置索引
quadkey_length = 25  # QuadKey精度级别
n = 6  # n-gram长度，根据作者参数设置，设为6

# 示例经纬度坐标
latitude = 40.64508  # 纽约的纬度
longitude = -74.78452  # 纽约的经度

# Step 1: 经纬度转 QuadKey
quadkey = latlng2quadkey(latitude, longitude, quadkey_length)
print("QuadKey:", quadkey)

# Step 2: 生成 QuadKey 的 n-grams 特征
def get_all_permutations_dict(length):
    # 使用字符 ['0', '1', '2', '3'] 生成所有可能的 n-grams 字符串组合
    characters = ['0', '1', '2', '3']
    all_permutations = [''.join(p) for p in product(characters, repeat=length)]
    premutation_dict = dict(zip(all_permutations, range(len(all_permutations))))
    return premutation_dict

# 使用新的函数生成6-grams的所有组合
permutations_dict = get_all_permutations_dict(n)

# 将 QuadKey 转换为 n-grams 序列
quadkey_ngrams = [''.join(x) for x in ngrams(quadkey, n)]
quadkey_ngrams = [permutations_dict.get(each, 0) for each in quadkey_ngrams]  # 将每个 n-gram 转换为字典中的索引
quadkey_ngrams_tensor = torch.tensor(quadkey_ngrams, dtype=torch.long).unsqueeze(0)  # 转换为张量并增加批次维度
print("QuadKey n-grams:", quadkey_ngrams)

# Step 3: 使用 GPSEmbeddings 和 GPSEncoder 处理地理位置嵌入
# 创建 GPSEmbeddings 实例
gps_embed_model = GPSEmbeddings(num_gps, gps_embed_dim)

# 加载预训练的 GPS 嵌入 (npy 文件)
gps_embeddings_np = np.load('/home/liuzhao/demo/LLM4POI/gps_process/result_embedding/gps/gps_embeddings.npy')
gps_embed_model.gps_embedding.weight.data = torch.tensor(gps_embeddings_np, dtype=torch.float32)

# 初始化 GPSEncoder 实例
nhead = 1  # 注意力头数量
nhid = 2 * gps_embed_dim  # FFN隐藏层维度
nlayers = 2  # Transformer层数
dropout = 0.3  # Dropout率
gps_encoder = GPSEncoder(gps_embed_dim, nhead, nhid, nlayers, dropout)

# 加载已训练的编码器参数
gps_encoder.load_state_dict(torch.load('/home/liuzhao/demo/LLM4POI/gps_process/result_embedding/gps/gps_encoder.pth', map_location='cpu'))
gps_encoder.eval()

# 生成 GPS 嵌入
gps_embeddings = gps_embed_model(quadkey_ngrams_tensor)

# Step 4: 使用 GPSEncoder 进行编码并查看输出
with torch.no_grad():
    output = gps_encoder(gps_embeddings)
    print("输出结果:", output)