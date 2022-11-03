from torch import nn

from models.definitions.backbone.vgg import vgg
from models.definitions.backbone.resnet import resnet50
from models.definitions.backbone.resnet_pretrained import resnext50_32x4d


class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()
        if 'vgg' in backbone:
            self.model = vgg(backbone, **kwargs)
        elif backbone == 'resnet50':
            self.model = resnet50(**kwargs)
        elif backbone == 'resnext50':
            self.model = resnext50_32x4d(pretrained=True)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != "last_conv_1x1":
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
