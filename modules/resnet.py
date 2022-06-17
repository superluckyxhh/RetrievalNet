from xml.dom import InvalidAccessErr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNet(nn.Module):
    def __init__(self, name, pretrained=False):
        super().__init__()
        
        if name == 'resnet152':
            resnet = torchvision.models.resnet152(pretrained=pretrained)

        elif name == 'resnet50':
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        
        elif name == 'resnet101':
            resnet = torchvision.models.resnet101(pretrained=pretrained)
        
        else:
            raise ValueError('Invalid backbone name')
    
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x3, x4