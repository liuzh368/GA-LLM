import torch
import numpy as np
from gps_model import GPSEncoder, GPSEmbeddings
from quad_key_encoder import latlng2quadkey
from itertools import product
from nltk import ngrams


class GeoEncoder:
    def __init__(self, gps_embed_dim=128, num_gps=4096, quadkey_length=25, n=6,
                 gps_embeddings_path='/home/liuzhao/demo/LLM4POI/gps_process/result_embedding/gps/gps_embeddings.npy',
                 gps_encoder_path='/home/liuzhao/demo/LLM4POI/gps_process/result_embedding/gps/gps_encoder.pth'):
        # 初始化参数
        self.gps_embed_dim = gps_embed_dim
        self.num_gps = num_gps
        self.quadkey_length = quadkey_length
        self.n = n

        # 加载嵌入和编码器
        self.gps_embed_model = GPSEmbeddings(self.num_gps, self.gps_embed_dim)
        gps_embeddings_np = np.load(gps_embeddings_path)
        self.gps_embed_model.gps_embedding.weight.data = torch.tensor(gps_embeddings_np, dtype=torch.float32)

        nhead = 1
        nhid = 2 * self.gps_embed_dim
        nlayers = 2
        dropout = 0.3
        self.gps_encoder = GPSEncoder(self.gps_embed_dim, nhead, nhid, nlayers, dropout)
        self.gps_encoder.load_state_dict(torch.load(gps_encoder_path, map_location='cpu'))
        self.gps_encoder.eval()

        # 生成 n-gram 字典
        self.permutations_dict = self.get_all_permutations_dict(self.n)

    def get_all_permutations_dict(self, length):
        characters = ['0', '1', '2', '3']
        all_permutations = [''.join(p) for p in product(characters, repeat=length)]
        permutation_dict = dict(zip(all_permutations, range(len(all_permutations))))
        return permutation_dict

    def generate_embedding(self, latitude, longitude):
        # Step 1: 经纬度转 QuadKey
        quadkey = latlng2quadkey(latitude, longitude, self.quadkey_length)

        # Step 2: 将 QuadKey 转换为 n-grams 序列
        quadkey_ngrams = [''.join(x) for x in ngrams(quadkey, self.n)]
        quadkey_ngrams = [self.permutations_dict.get(each, 0) for each in quadkey_ngrams]
        quadkey_ngrams_tensor = torch.tensor(quadkey_ngrams, dtype=torch.long).unsqueeze(0)

        # Step 3: 生成 GPS 嵌入
        gps_embeddings = self.gps_embed_model(quadkey_ngrams_tensor)

        # Step 4: 使用 GPSEncoder 进行编码并输出
        with torch.no_grad():
            output = self.gps_encoder(gps_embeddings)
        return output


# 使用示例
geo_encoder = GeoEncoder()
latitude = 40.7128  # 纽约的纬度
longitude = -74.0060  # 纽约的经度
embedding = geo_encoder.generate_embedding(latitude, longitude)
print("Embedding:", embedding)