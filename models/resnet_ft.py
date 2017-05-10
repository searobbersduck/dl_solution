import torch
import torch.nn as nn
import math
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

resnet_ft_model_names = [
    '18',
    '34',
    '50',
    '101',
    '152',
]

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_FT(nn.Module):
    def __init__(self, models='18', pretrained=False, pretrained_model=None, downsample=False):
        super(ResNet_FT,self).__init__()
        self.model = self.getModel(models)
        if self.model is None:
            print('ResNet Fine Tune model failed!')
            return
        if pretrained:
            self.model.load_state_dict(torch.load(pretrained_model))
            for param in self.model.parameters():
                param.requires_grad = False

        if not downsample:
            self.model.avgpool = nn.AvgPool2d(14)
        self.fc = nn.Linear(1000, 5)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

    def getModel(self, modelsname):
        if modelsname == '18':
            return models.ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        elif modelsname == '34':
            return models.ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000)
        elif modelsname == '50':
            return models.ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)
        elif modelsname == '101':
            return models.ResNet(Bottleneck, [3, 4, 23, 3], num_classes=1000)
        elif modelsname == '152':
            return models.ResNet(Bottleneck, [3, 8, 36, 3], num_classes=1000)
        else:
            return None