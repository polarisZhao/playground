''' 
VGG11/13/16/19 in Pytorch.

Author: zhaozhichao
Paper: Very Deep Convolutional Network for Large-Scale Image Recognition
Modified from https://github.com/pytorch/vision.git
'''
import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),  # change out channel
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 10),  # num of class
        ) 

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),  # add batchnorm 
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def VGG11():
    return VGG('VGG11')

def VGG13():
    return VGG('VGG13')

def VGG16():
    return VGG('VGG16')

def VGG19():
    return VGG('VGG19')

# net = VGG19()
# x = torch.randn(2, 3, 224, 224)
# print(net(x))
