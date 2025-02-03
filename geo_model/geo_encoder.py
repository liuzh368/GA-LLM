# geo_encoder.py
import torch
import torch.nn as nn
import numpy as np
from geo_model.gps_model import GPSEncoder, GPSEmbeddings
from geo_model.quad_key_encoder import latlng2quadkey
from itertools import product
from nltk import ngrams


class GeoEncoder(nn.Module):
    def __init__(self, gps_embed_dim=4096, num_gps=4096, quadkey_length=25, n=6,
                 gps_embeddings_path=None,
                 gps_encoder_path=None):
        super(GeoEncoder, self).__init__()  # 调用父类的构造函数
        # 初始化参数
        self.gps_embed_dim = gps_embed_dim
        self.num_gps = num_gps
        self.quadkey_length = quadkey_length
        self.n = n

        # 创建 GPSEmbeddings 实例
        self.gps_embed_model = GPSEmbeddings(self.num_gps, self.gps_embed_dim)

        # 如果提供了预训练的 GPS 嵌入，则加载
        if gps_embeddings_path:
            gps_embeddings_np = np.load(gps_embeddings_path)
            self.gps_embed_model.gps_embedding.weight.data = torch.tensor(gps_embeddings_np, dtype=torch.float32)

        # 初始化 GPSEncoder 实例
        nhead = 1
        nhid = 2 * self.gps_embed_dim
        nlayers = 2
        dropout = 0.3
        self.gps_encoder = GPSEncoder(self.gps_embed_dim, nhead, nhid, nlayers, dropout)

        print("Current GPSEncoder model structure:")
        for name, param in self.gps_encoder.named_parameters():
            print(name, param.shape)
        # 如果提供了预训练的编码器参数，则加载
        if gps_encoder_path:
            checkpoint_state_dict = torch.load(gps_encoder_path, map_location='cpu')
            print("Loaded checkpoint state_dict structure:")
            for name, param in checkpoint_state_dict.items():
                print(name, param.shape)
            # 进行加载
            self.gps_encoder.load_state_dict(checkpoint_state_dict, strict=False)
            print("load pre_gps_encoder!")
        self.gps_encoder.eval()

        # 生成 n-gram 字典
        self.permutations_dict = self.get_all_permutations_dict(self.n)

    def get_all_permutations_dict(self, length):
        characters = ['0', '1', '2', '3']
        all_permutations = [''.join(p) for p in product(characters, repeat=length)]
        permutation_dict = dict(zip(all_permutations, range(len(all_permutations))))
        return permutation_dict

    def generate_embedding(self, latitude, longitude):
        # 获取模型的设备
        device = next(self.parameters()).device

        # Step 1: 经纬度转 QuadKey
        quadkey = latlng2quadkey(latitude, longitude, self.quadkey_length)

        # Step 2: 将 QuadKey 转换为 n-grams 序列
        quadkey_ngrams = [''.join(x) for x in ngrams(quadkey, self.n)]
        quadkey_ngrams = [self.permutations_dict.get(each, 0) for each in quadkey_ngrams]
        quadkey_ngrams_tensor = torch.tensor(quadkey_ngrams, dtype=torch.long).unsqueeze(0).to(device)

        # Step 3: 生成 GPS 嵌入
        gps_embeddings = self.gps_embed_model(quadkey_ngrams_tensor)

        # Step 4: 使用 GPSEncoder 进行编码并输出
        output = self.gps_encoder(gps_embeddings)
        return output  # 输出维度为 gps_embed_dim