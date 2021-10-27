from torch import nn

from models.definitions.backbone.vgg import vgg19
from models.definitions.backbone.resnet import resnet50


class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()
        if backbone == "vgg19":
            self.model = vgg19(**kwargs)
        elif backbone == 'resnet50':
            self.model = resnet50(**kwargs)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != "last_conv_1x1":
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
