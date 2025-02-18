import torch 
from torch import nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, emb_dim, patch_size):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        B, emb_dim, H_p, N_p = x.shape
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x