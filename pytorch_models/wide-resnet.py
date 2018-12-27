''' 
Wide-ResNet in Pytorch.

Author: zhaozhichao
Paper: 

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class wide_basic(nn.Module): 
    expansion = 1

    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(wide_basic, self).__init__() 
        self.downsample = (in_channels != out_channels)  
        self.strides = 2 if self.downsample else 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.strides, padding=1, bias=False)
        self.dropout = nn.Dropout(p=drop_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=self.strides, bias=False))           

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.downsample:
            x = self.shortcut(x)
        out += x
        return F.relu(out)

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3] * 9, num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_planes, planes, dropout_rate))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

net=Wide_ResNet(28, 10, 0.3, 10)
y = net(torch.randn(1,3,224,224))
print(y.size())
