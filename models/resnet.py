'''
ResNet in PyTorch.
Author: zhaozhichao

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385


'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    expansion=1
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__() 
        self.downsample = (in_channels != out_channels)  
        self.strides = 2 if self.downsample else 1           
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.strides, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=self.strides, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels))           

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            x = self.shortcut(x)
        out += x
        return F.relu(out)
    
    
class Bottleneck(nn.Module):
    expansion = 4  
    
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.downsample = (in_channels != self.expansion * out_channels)  
        self.strides = 2 if (in_channels != out_channels) else 1  
        self.conv1 = nn.Conv2d(self.expansion*in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=self.strides, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)            
        if self.downsample:  
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.expansion*in_channels, self.expansion*out_channels, 
                          kernel_size=1, stride=self.strides, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            x = self.shortcut(x)
        out += x
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64/block.expansion, 64, num_blocks[0])       
        self.layer2 = self._make_layer(block, 64, 128, num_blocks[1])
        self.layer3 = self._make_layer(block, 128, 256, num_blocks[2])
        self.layer4 = self._make_layer(block, 256, 512, num_blocks[3]) 
        self.fc = nn.Linear(512 * block.expansion * 1, num_classes)  #  change inchannels and num of classes 
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(block(in_channels, out_channels))         
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layer0(x)        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        print(x.size()[1] / 512)
        return self.fc(x)

def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(ResidualBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

# net = ResNet50()
# y = net(torch.randn(1, 3, 224, 224))
# print(y)

