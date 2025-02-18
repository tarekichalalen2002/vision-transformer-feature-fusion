import torch
from torch import nn
from torch.nn import functional as F

class VisionTransformer(nn.Module):
    def __init__(self, emb_dim, depth, n_heads, mlp_dim, dropout=0.1):
        super(VisionTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = emb_dim,
            nhead = n_heads,
            dim_feedforward = mlp_dim,
            dropout=dropout,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim = 1)
        return x