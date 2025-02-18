from torch import nn
from torch.nn import functional as F

class LBPFeatureExtractor(nn.Module):
    def __init__(self):
        super(LBPFeatureExtractor, self).__init__() 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        x_gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        x = F.relu(self.bn1(self.conv1(x_gray)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x