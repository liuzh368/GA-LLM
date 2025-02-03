# geo_model/fusion_geo_encoder.py

import math
import torch
import torch.nn as nn
import numpy as np
from itertools import product
from nltk import ngrams

EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
MinLongitude = -180
MaxLongitude = 180

def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)

def map_size(levelOfDetail):
    return 256 << levelOfDetail

def latlng2pxy(latitude, longitude, levelOfDetail):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, MinLongitude, MaxLongitude)

    x = (longitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    ms = map_size(levelOfDetail)
    pixelX = int(clip(x * ms + 0.5, 0, ms - 1))
    pixelY = int(clip(y * ms + 0.5, 0, ms - 1))
    return pixelX, pixelY

def pxy2txy(pixelX, pixelY):
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY

def txy2quadkey(tileX, tileY, levelOfDetail):
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))
    return ''.join(quadKey)

def latlng2quadkey(lat, lng, level=25):
    pixelX, pixelY = latlng2pxy(lat, lng, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY, level)

# --------------------------
# LearnableFourierPositionalEncoding
# --------------------------
class LearnableFourierPositionalEncoding(nn.Module):
    """
    输入: x.shape = [N, G, M], G=1, M=25(QuadKey长度)
    输出: [N, D]
    """
    def __init__(self, G, M, F_dim, H_dim, D, gamma=10.0):
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        # x: [N, G, M]
        device = self.Wr.weight.device
        dtype  = self.Wr.weight.dtype
        x = x.to(device=device, dtype=dtype)

        N, G, M = x.shape
        projected = self.Wr(x)  # [N, G, F_dim//2], bf16
        cosines = torch.cos(projected)
        sines   = torch.sin(projected)
        F = (1 / math.sqrt(self.F_dim)) * torch.cat([cosines, sines], dim=-1)  # [N, G, F_dim], bf16

        Y = self.mlp(F)  # [N, G, D//G]
        PEx = Y.reshape(N, self.D)
        return PEx

class GPSPositionalEncoder(nn.Module):
    """
    将QuadKey(25维)投影到D维 (例如128 或 4096)
    """
    def __init__(self, M=25, F_dim=256, H_dim=128, D=128, gamma=10.0):
        super().__init__()
        self.G = 1
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        self.pos_encoder = LearnableFourierPositionalEncoding(
            G=self.G,
            M=self.M,
            F_dim=self.F_dim,
            H_dim=self.H_dim,
            D=self.D,
            gamma=self.gamma
        )

    def forward(self, latitude, longitude):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        # 创建QuadKey(25维)
        quadkey = latlng2quadkey(latitude, longitude, level=self.M)
        quadkey_digits = [int(c) for c in quadkey]  # len=25
        x = torch.tensor(quadkey_digits, dtype=dtype).unsqueeze(0).unsqueeze(0).to(device)

        pex = self.pos_encoder(x)  # [1, D], bf16
        return pex  # [1,D], 同device/dtype

class GPSEmbeddings(nn.Module):
    def __init__(self, num_gps, embedding_dim):
        super().__init__()
        self.gps_embedding = nn.Embedding(num_embeddings=num_gps, embedding_dim=embedding_dim)

    def forward(self, gps_idx):
        gps_idx = gps_idx.to(self.gps_embedding.weight.device)
        embed = self.gps_embedding(gps_idx)
        return embed

class GPSEncoder(nn.Module):
    """
    小型Transformer, 最终输出 [batch, embed_size]
    """
    def __init__(self, embed_size=128, nhead=1, nhid=256, nlayers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.embed_size = embed_size
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, src):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        src = src.to(device=device, dtype=dtype)
        src = src * math.sqrt(self.embed_size)
        x = self.transformer_encoder(src)    # [batch, seq_len, embed_size], bf16
        x = torch.mean(x, dim=1)             # [batch, embed_size]
        return self.norm(x)

class FusionGeoEncoder4096(nn.Module):
    """
    - 先 n-gram + Embedding(128) + Transformer => [1,128]
    - 再 QuadKey整体可学习傅里叶 => [1,128]
    - 拼接 => [1,256] => Linear => [1,4096]
    """
    def __init__(self,
                 gps_embed_dim=128,
                 quadkey_length=25,
                 n=6,
                 vocab_size=4096,
                 fourier_output_dim=128,
                 final_dim=4096,
                 dropout=0.1):
        super().__init__()
        self.gps_embed_dim = gps_embed_dim
        self.quadkey_length = quadkey_length
        self.n = n

        self.permutations_dict = self._build_permutations_dict(self.n)

        self.gps_embed_model = GPSEmbeddings(num_gps=vocab_size, embedding_dim=gps_embed_dim)
        self.gps_encoder = GPSEncoder(
            embed_size=gps_embed_dim,
            nhead=1,
            nhid=2*gps_embed_dim,
            nlayers=2,
            dropout=dropout
        )
        self.fourier_encoder = GPSPositionalEncoder(
            M=self.quadkey_length,
            F_dim=256,
            H_dim=128,
            D=fourier_output_dim,
            gamma=10.0
        )

        self.final_linear = nn.Linear(gps_embed_dim + fourier_output_dim, final_dim)

    def _build_permutations_dict(self, length):
        chars = ['0','1','2','3']
        all_permutations = [''.join(p) for p in product(chars, repeat=length)]
        return dict(zip(all_permutations, range(len(all_permutations))))

    def forward(self, latitude, longitude):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        # Step A: n-gram => [1, seq_len]
        quadkey = latlng2quadkey(latitude, longitude, self.quadkey_length)
        quadkey_ngrams = [''.join(x) for x in ngrams(quadkey, self.n)]
        qk_idx = [self.permutations_dict.get(ng, 0) for ng in quadkey_ngrams]
        qk_idx_tensor = torch.tensor(qk_idx, dtype=torch.long).unsqueeze(0).to(device)

        token_embeddings = self.gps_embed_model(qk_idx_tensor)   # [1, seq_len, 128] bf16
        token_embeddings = token_embeddings.to(dtype=dtype)

        emb_128 = self.gps_encoder(token_embeddings)  # [1,128], bf16

        # Step B: Fourier => [1,128], bf16
        fourier_128 = self.fourier_encoder(latitude, longitude)
        fourier_128 = fourier_128.to(device=device, dtype=dtype)

        # 拼接 => [1,256]
        fused = torch.cat([emb_128, fourier_128], dim=-1)  # [1,256], bf16
        fused = fused.to(device=device, dtype=dtype)

        final_4096 = self.final_linear(fused)  # [1,4096], bf16
        return final_4096

if __name__ == "__main__":
    model_4096 = FusionGeoEncoder4096()

    latitude = 40.7128
    longitude = -74.0060
    emb_4096 = model_4096(latitude, longitude)
    print("Final output shape:", emb_4096.shape)
    print(emb_4096)