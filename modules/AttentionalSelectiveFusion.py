import torch 
from torch import nn

class AttentionalSelectiveFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionalSelectiveFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        attention = self.sigmoid(self.conv(x))
        out = attention * feat1 + (1 - attention) * feat2   
        return out
    
