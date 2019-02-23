import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3, grouped_conv=True, combine='add'):        
        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        if self.combine == 'add':
            self.depthwise_stride = 1
        else:
            self.depthwise_stride = 2
            self.out_channels -= self.in_channels
        self.first_1x1_groups = self.groups if grouped_conv else 1
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(self.in_channels, self.bottleneck_channels,
            self.first_1x1_groups, relu=True)
        self.depthwise_conv3x3 = nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, kernel_size=3, 
            stride=self.depthwise_stride, padding=1, bias=True, groups=self.bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(self.bottleneck_channels, self.out_channels, self.groups,
            relu=False)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups, relu=False):
        modules = OrderedDict()
        modules['conv1x1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)
        modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        return nn.Sequential(modules)

    def forward(self, x):
        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)       
        if self.combine == 'concat':
            residual = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
            out = torch.cat((residual, out), 1)
        else:
            out += residual
        return F.relu(out)


class ShuffleNet(nn.Module):
    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        super(ShuffleNet, self).__init__()
        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        else:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]       
        self.conv1 =  nn.Conv2d(in_channels, self.stage_out_channels[1], kernel_size=3, 
            stride=2, padding=1, bias=True, groups=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(2)
        self.stage3 = self._make_stage(3)
        self.stage4 = self._make_stage(4)
        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, num_classes)

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage)

        grouped_conv = stage > 2
        first_module = ShuffleUnit(self.stage_out_channels[stage-1], self.stage_out_channels[stage],
            groups=self.groups, grouped_conv=grouped_conv, combine='concat')
        modules[stage_name+"_0"] = first_module
        for i in range(self.stage_repeats[stage-2]):
            name = stage_name + "_{}".format(i+1)
            module = ShuffleUnit(self.stage_out_channels[stage], self.stage_out_channels[stage],
                groups=self.groups, grouped_conv=True, combine='add')
            modules[name] = module

        return nn.Sequential(modules)


    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, x.data.size()[-2:])
        x = x.view(x.size(0), -1) #
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

model = ShuffleNet()
inputs = torch.rand(2, 3, 224, 224)
print(model(inputs).size())
