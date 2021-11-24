import torch
from torch import nn
from torchvision import models


class VGG(nn.Module):
    def __init__(self, ss, ks, hidden, pretrained=True, dropout=0.1):
        super(VGG, self).__init__()
        cnn = models.vgg19_bn(pretrained=pretrained)
        pool_idx = 0
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn.features[i] = torch.nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1

        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

    def forward(self, x):
        conv = self.features(x)             # (N, C, H, W)
        conv = self.dropout(conv)           # (N, C, H, W)
        conv = self.last_conv_1x1(conv)     # (N, C, H, W)
        conv = conv.transpose(-1, -2)       # (N, C, W, H)
        conv = conv.flatten(2)              # (N, C, W*H)
        conv = conv.permute(-1, 0, 1)       # (W*H, N, C)
        return conv


def vgg19(ss, ks, hidden, pretrained=True, dropout=0.1):
    return VGG(ss, ks, hidden, pretrained, dropout)
