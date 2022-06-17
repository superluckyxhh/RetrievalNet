import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m ,nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.conv2(x)
        x = self.drop(x)
        
        return x