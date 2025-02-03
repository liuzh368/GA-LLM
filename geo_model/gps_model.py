# gps_model.py
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GPSEmbeddings(nn.Module):
    def __init__(self, num_gps, embedding_dim):
        super(GPSEmbeddings, self).__init__()
        self.gps_embedding = nn.Embedding(
            num_embeddings=num_gps,
            embedding_dim=embedding_dim
        )

    def forward(self, gps_idx):
        embed = self.gps_embedding(gps_idx)
        return embed

class GPSEncoder(nn.Module):
    def __init__(self, embed_size, nhead, nhid, nlayers, dropout):
        super(GPSEncoder, self).__init__()
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, src):
        src = src * math.sqrt(self.embed_size)
        x = self.transformer_encoder(src)
        x = torch.mean(x, dim=1)  # 对序列长度维度进行平均
        return self.norm(x)