import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class AffineTransform(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.transform_affine_matrix = nn.Conv2d(in_planes, 4, kernel_size=3, padding=1, bias=True)
        self.transform_xy = nn.Conv2d(in_planes, 2, kernel_size=3, padding=1, bias=True)
        
        self.conv1 = nn.Conv2d(2, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 512, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.conv4 = nn.Conv2d(512, out_planes, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(out_planes)
        
        
    def forward(self, x):
        b, c, h, w = x.size()
        residual = x
        # [b*h*w, 2, 2]
        affine_matrix = self.transform_affine_matrix(x)
        affine_matrix = affine_matrix.permute(0, 2, 3, 1).reshape((b*h*w, 2, 2))
        # [b*h*w, 2, 1]
        xy = self.transform_xy(x).permute(0, 2, 3, 1).reshape((b*h*w, 2)).unsqueeze(-1)
        affine_xy = torch.einsum('nab,ncd->nad', affine_matrix, xy)
        # affine_xy = torch.bmm(affine_matrix, xy)
        affine_xy = affine_xy.squeeze(-1).reshape((b, h, w, 2))
        affine_xy = affine_xy.permute((0, 3, 1, 2))
        
        affine_feats = self.bn1(self.conv1(affine_xy))
        affine_feats = self.bn2(self.conv2(affine_feats))
        affine_feats = self.bn3(self.conv3(affine_feats))
        affine_feats = self.bn4(self.conv4(affine_feats))
        
        return affine_feats + residual