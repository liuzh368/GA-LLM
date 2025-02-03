import numpy as np
import torch
import torch.nn as nn


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        """
        可学习的傅里叶特征位置编码类
        基于论文 https://arxiv.org/pdf/2106.02795.pdf (算法1)
        实现算法1：计算多维位置的傅里叶特征位置编码
        输入形状为 [N, G, M] 的张量
        参数：
        :param G: 位置组的数量（不同组之间位置独立）
        :param M: 每个点的 M 维位置值
        :param F_dim: 傅里叶特征的维度
        :param H_dim: 隐藏层的维度
        :param D: 最终位置编码的维度
        :param gamma: 用于初始化 Wr 的参数
        """
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


if __name__ == '__main__':
    # 参数设置
    G = 1  # 位置分组为1（经纬度属于同一组）
    M = 2  # 经纬度为二维输入
    F_dim = 1024  # 傅里叶特征维度
    H_dim = 512  # 隐藏层维度
    D = 4096  # 输出维度（4096维向量）
    gamma = 10  # 初始化参数

    # 经纬度输入
    latitude = 40.64508  # 纬度
    longitude = -74.78452  # 经度
    coordinates = torch.tensor([[latitude, longitude]])  # 输入形状为 [1, 2]

    # 调整形状为 [N, G, M]
    x = coordinates.unsqueeze(1)  # [1, 1, 2]

    # 初始化模型
    enc = LearnableFourierPositionalEncoding(G, M, F_dim, H_dim, D, gamma)

    # 计算位置编码
    pex = enc(x)

    # 输出结果
    print("输出向量形状:", pex.shape)
    print("输出向量:", pex)