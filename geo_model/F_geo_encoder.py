# combined_model.py

import math
import torch
import torch.nn as nn
import numpy as np

# 定义 LearnableFourierPositionalEncoding 类
class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        # 用于学习傅里叶特征的投影矩阵 (公式2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)

        # 多层感知机 (MLP) (公式6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G)
        )

        self.init_weights()

    def init_weights(self):
        """初始化权重，将 Wr 初始化为正态分布"""
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        生成位置编码
        :param x: 输入张量，形状为 [N, G, M]，表示 N 个位置，每个位置分为 G 组，每组 M 维。
        :return: 输出位置编码，形状为 [N, D]
        """
        N, G, M = x.shape
        # 第一步：计算傅里叶特征 (公式2)
        projected = self.Wr(x)  # 通过 Wr 投影
        cosines = torch.cos(projected)  # 余弦部分
        sines = torch.sin(projected)  # 正弦部分
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)  # 拼接并归一化

        # 第二步：通过 MLP 处理傅里叶特征 (公式6)
        Y = self.mlp(F)

        # 第三步：调整输出形状
        PEx = Y.reshape((N, self.D))
        return PEx

# 定义用于转换经纬度到 QuadKey 的函数
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

    mapSize = map_size(levelOfDetail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY

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

def pxy2txy(pixelX, pixelY):
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY

def latlng2quadkey(lat, lng, level):
    pixelX, pixelY = latlng2pxy(lat, lng, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY, level)

# 定义 GPSPositionalEncoder 类
class GPSPositionalEncoder(nn.Module):
    def __init__(self, G=1, M=25, F_dim=1024, H_dim=512, D=4096, gamma=10):
        super().__init__()
        self.G = G
        self.M = M  # QuadKey 长度，也即 M 维输入
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
        # 将经纬度转换为 QuadKey
        level = self.M  # 假设 M 就是 QuadKey 的长度
        quadkey = latlng2quadkey(latitude, longitude, level)

        # 将 QuadKey 字符串映射为数值向量
        # 将 '0' -> 0, '1' -> 1, '2' -> 2, '3' -> 3
        quadkey_digits = [int(c) for c in quadkey]
        # 转换为张量
        x = torch.tensor(quadkey_digits, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 形状 [1, 1, M]

        # 通过位置编码器
        pex = self.pos_encoder(x)

        return pex  # 输出形状 [1, D]