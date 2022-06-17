import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from timm.models.layers import DropPath, trunc_normal_

from modules.norm import LayerNormChannel, GroupNorm
from modules.mlp import MLP


def elu_feature_map(x):
    return F.elu(x) + 1


class Attention(nn.Module):
    def __init__(self, dim, head_dim=32, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % head_dim == 0, "Feature Dim Should be Divisble by Head Dim"
        self.head_dim = head_dim
        self.num_head = dim // head_dim
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x0, x1):
        b, c, h, w = x0.size()
        n = h * w
        x0 = torch.flatten(x0, start_dim=2).transpose(-2, -1)
        x1 = torch.flatten(x1, start_dim=2).transpose(-2, -1)
        
        # [3, b, num_head, n, head_dim]
        q = self.q(x0).reshape(b, n, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x1).reshape(b, n, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x1).reshape(b, n, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-2, -1).reshape(b, c, h, w)
        
        return x       
            
        
class TransformerUnit(nn.Module):
    def __init__(self, dim, token_mixer, 
                 mlp_ratio=4, act_layer=nn.GELU,
                 norm_layer=LayerNormChannel, drop=0., 
                 drop_path=0.
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        self.token_mixer = token_mixer
        self.norm3 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, src):
        x = self.norm1(x)
        src = self.norm2(src)
        
        x = x + self.drop_path(self.token_mixer(x, src))
        x = x + self.drop_path(self.mlp(self.norm3(x)))  
        
        return x      


class LinearAttention(nn.Module):
    def __init__(self, dim, head_dim=32, qkv_bias=False, attn_drop=0., proj_drop=0., eps=1e-5):
        super().__init__()
        assert dim % head_dim == 0, "Feature Dim Should be Divisble by Head Dim"
        self.head_dim = head_dim
        self.num_head = dim // head_dim
        
        self.proj = nn.Linear(dim, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.elu_feature_map = elu_feature_map
        self.eps = eps
    
    def forward(self, q, k, v, q_mask=None, k_mask=None):
        b, c, h, w = q.size()
        n = h * w
        q = torch.flatten(q, start_dim=2).transpose(-2, -1)
        k = torch.flatten(k, start_dim=2).transpose(-2, -1)
        v = torch.flatten(v, start_dim=2).transpose(-2, -1)

        # [3, b, n, num_head, head_dim]
        q = self.q(q).view(b, -1, self.num_head, self.head_dim)
        k = self.k(k).view(b, -1, self.num_head, self.head_dim)
        v = self.v(v).view(b, -1, self.num_head, self.head_dim)
       
        Q = self.elu_feature_map(q)
        K = self.elu_feature_map(k)
        
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if k_mask is not None:
            K = K * k_mask[:, :, None, None]
            v = v * k_mask[:, :, None, None]
            
        # [b, n, head_dim, head_dim]
        KV = torch.einsum("bnhd,bnhm->bnmd", K, v)
        # [b, n, num_head]
        Z = 1 / (torch.einsum("bnhd,bhd->bnh", Q, K.sum(dim=1)) + self.eps)
        # [b, n, num_head, head_dim]
        V = torch.einsum("bnhd,bnmd,bnh->bnhm", Q, KV, Z)
        V = V.contiguous().view(b, n, -1)
        
        V = self.proj(V)
        x = V.transpose(-2, -1).reshape(b, c, h, w)
        
        return x
        
        
class LinearTransformerUnit(nn.Module):
    def __init__(self, dim, token_mixer, mlp_ratio=2, 
                 drop=0., act_layer=nn.GELU, 
                 norm_layer=LayerNormChannel
    ):
        super().__init__()
        self.token_mixer = token_mixer
        self.dim = dim
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = MLP(in_features=dim*2, hidden_features=mlp_hidden_dim,
                       out_features=dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x, src, x_mask=None, src_mask=None):
        b, c, h, w = x.size()
        msg = self.norm1(self.token_mixer(x, src, src, q_mask=x_mask, k_mask=src_mask))
        msg = torch.cat([x, msg], dim=1)
        msg = self.norm2(self.mlp(msg))
        
        return x + msg