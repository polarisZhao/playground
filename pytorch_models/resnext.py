'''
ResNet in PyTorch.
Author: zhaozhichao
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
    
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNextResidualBlock(nn.Module):
    expansion=1
    
    def __init__(self, in_channels, out_channels, num_group=32):
        super(ResNextResidualBlock, self).__init__() 
        self.downsample = (in_channels != out_channels)  
        self.strides = 2 if self.downsample else 1           
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.strides, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(self.expansion*out_channels)
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=self.strides, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels))           

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.downsample:
            x = self.shortcut(x)
        out += x
        return F.relu(out)
    
     
class ResNextBottleneck(nn.Module):
    expansion = 4  
    
    def __init__(self, in_channels, out_channels, num_group=32):
        super(ResNextBottleneck, self).__init__()
        self.downsample = (in_channels != out_channels)  
        self.strides = 2 if (in_channels != out_channels) else 1         
        self.conv1 = nn.Conv2d(self.expansion*in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=self.strides,
                               padding=1, bias=False, groups=num_group)
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
       
class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, num_group=32):
        super(ResNeXt, self).__init__()
        self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64//block.expansion, 64, num_blocks[0], num_group)       
        self.layer2 = self._make_layer(block, 64, 128, num_blocks[1], num_group)
        self.layer3 = self._make_layer(block, 128, 256, num_blocks[2], num_group)
        self.layer4 = self._make_layer(block, 256, 512, num_blocks[3], num_group) 
        self.fc = nn.Linear(512 * block.expansion * 1, num_classes)  #  change inchannels and num of classes 
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks, num_group):
        layers = []
        layers.append(block(in_channels, out_channels, num_group))         
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, num_group))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layer0(x)        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        # print(x.size()[1] / 512)
        return self.fc(x)

def ResNeXt18():
    return ResNeXt(ResNextResidualBlock, [2, 2, 2, 2])

def ResNeXt34():
    return ResNeXt(ResNextResidualBlock, [3, 4, 6, 3])

def ResNeXt50():
    return ResNeXt(ResNextBottleneck, [3, 4, 6, 3])

def ResNeXt101():
    return ResNeXt(ResNextBottleneck, [3, 4, 23, 3])

def ResNeXt152():
    return ResNeXt(ResNextBottleneck, [3, 8, 36, 3])

# net = ResNeXt152()
# y = net(torch.randn(2, 3, 224, 224))
# print(y.size())