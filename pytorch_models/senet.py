'''SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
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
        self.conv2 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*out_channels)
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=self.strides, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)) 
        # SE layers
        self.fc1 = nn.Conv2d(self.expansion*out_channels, self.expansion*out_channels//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(self.expansion*out_channels//16, self.expansion*out_channels, kernel_size=1)         

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        if self.downsample:
            x = self.shortcut(x)
        out += x
        return F.relu(out)

class PreResidualBlock(nn.Module):
    expansion=1
    
    def __init__(self, in_channels, out_channels):
        super(PreResidualBlock, self).__init__() 
        self.downsample = (in_channels != out_channels)  
        self.strides = 2 if self.downsample else 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.strides, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=self.strides, bias=False))
        # SE layers
        self.fc1 = nn.Conv2d(self.expansion*out_channels, self.expansion*out_channels//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(self.expansion*out_channels//16, self.expansion*out_channels, kernel_size=1)                  

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        if self.downsample:
            x = self.shortcut(x)
        out += x
        return F.relu(out)

class Bottleneck(nn.Module):
    expansion = 4  
    
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.downsample = (in_channels != out_channels)  
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
        # SE layers
        self.fc1 = nn.Conv2d(self.expansion*out_channels, self.expansion*out_channels//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(self.expansion*out_channels//16, self.expansion*out_channels, kernel_size=1)   

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        if self.downsample:
            x = self.shortcut(x)
        out += x
        return F.relu(out)

class PreBottleneck(nn.Module):
    expansion = 4  
    
    def __init__(self, in_channels, out_channels):
        super(PreBottleneck, self).__init__()
        self.downsample = (in_channels != out_channels)  
        self.strides = 2 if (in_channels != out_channels) else 1 
        self.bn1 = nn.BatchNorm2d(self.expansion*in_channels)
        self.conv1 = nn.Conv2d(self.expansion*in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=self.strides, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels) 
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=1, bias=False)           
        if self.downsample:  
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(self.expansion * in_channels),
                nn.Conv2d(self.expansion * in_channels, self.expansion*out_channels, 
                          kernel_size=1, stride=self.strides, bias=False))

        # SE layers
        self.fc1 = nn.Conv2d(self.expansion*out_channels, self.expansion*out_channels//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(self.expansion*out_channels//16, self.expansion*out_channels, kernel_size=1) 

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        if self.downsample:
            x = self.shortcut(x)
        out += x
        return F.relu(out)

        if self.downsample:
            x = self.shortcut(x)
        out += x
        return F.relu(out)

class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(SENet, self).__init__()
        self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64//block.expansion, 64, num_blocks[0])       
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
        # print(x.size()[1] / 512)
        return self.fc(x)


def SENet18():
    return SENet(ResidualBlock, [2, 2, 2, 2])

def SENet34():
    return SENet(ResidualBlock, [3, 4, 6, 3])

def SENet50():
    return SENet(Bottleneck, [3, 4, 6, 3])

def SENet101():
    return SENet(Bottleneck, [3, 4, 23, 3])

def SENet152():
    return SENet(Bottleneck, [3, 8, 36, 3])


net = SENet152()
y = net(torch.randn(2,3,224,224))
print(y.size())


