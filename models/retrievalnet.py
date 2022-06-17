import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from modules.resnet import ResNet
from modules.pool import GeMPool
from modules.transformers import Attention, TransformerUnit, LinearAttention, LinearTransformerUnit
from modules.affine_transform import AffineTransform
from modules.position_embedding import PositionEmbSin2D
from modules.norm import LayerNormChannel, GroupNorm, L2Norm
from modules.matching import MatchAlign
from core.config import cfg
        
        
class StructBranch(nn.Module):
    def __init__(self, in_planes=1024, hid_planes=512, desc_planes=128):
        super().__init__()
        self.atrous1 = nn.Conv2d(in_planes, hid_planes, kernel_size=3, dilation=6, padding=6)
        self.bn1 = nn.BatchNorm2d(hid_planes)
        self.relu1 = nn.ReLU()
        
        self.atrous2 = nn.Conv2d(in_planes, hid_planes, kernel_size=3, dilation=12, padding=12)
        self.bn2 = nn.BatchNorm2d(hid_planes)
        self.relu2 = nn.ReLU()

        # self.atrous3 = nn.Conv2d(in_planes2, hid_planes, kernel_size=3, dilation=18, padding=18)
        # self.bn2 = nn.BatchNorm2d(hid_planes)
        # self.relu2 = nn.ReLU()
        
        self.merge = nn.Conv2d(hid_planes*2, in_planes, kernel_size=1)
        self.merge_bn = nn.BatchNorm2d(in_planes)
        self.affline_transform = AffineTransform(in_planes, in_planes)
        
        self.positional_embedding = PositionEmbSin2D(in_planes)
        
        # Transformer
        # self.transformer_unit = TransformerUnit(in_planes, Attention(in_planes), mlp_ratio=1)
        
        # Linear Transformer
        self.transformer_unit = LinearTransformerUnit(in_planes, LinearAttention(in_planes), mlp_ratio=1)
    
    def forward(self, feat):
        _, c, h1, w1 = feat.size()
        atrous_feat1 = self.relu1(self.bn1(self.atrous1(feat)))
        atrous_feat2 = self.relu2(self.bn2(self.atrous2(feat)))
        feats = torch.cat([atrous_feat1, atrous_feat2], dim=1)
        
        feats_pos = self.positional_embedding(feats)
        feats = feats + feats_pos
        
        feats = self.merge(feats)
        feats = self.merge_bn(feats)
        feats = self.affline_transform(feats)
        # x = self.deform_conv(x)
        feats = self.transformer_unit(feats, feats)
        
        return feats


class FusionBranch(nn.Module):
    def __init__(self, mlp_ratio=1, locaL_planes=1024, global_planes=2048):
        super().__init__()
        self.local_conv = nn.Conv2d(locaL_planes, global_planes, 1)
        self.local_bn = nn.BatchNorm2d(global_planes)
        
        self.fg_gem_pool = GeMPool(gem_p=cfg.GEM.P, trainable=cfg.GEM.TRAIN)
        self.fl_gem_pool = GeMPool(gem_p=cfg.GEM.P, trainable=cfg.GEM.TRAIN)
        
        self.fg_norm = nn.LayerNorm(global_planes)
        self.fl_norm = nn.LayerNorm(global_planes)

        self.posemb_global = PositionEmbSin2D(global_planes)
        self.posemb_local = PositionEmbSin2D(global_planes)

        self.fusion_fg_fl_transformer = LinearTransformerUnit(global_planes, LinearAttention(global_planes), mlp_ratio=mlp_ratio)
        
    def forward(self, fg, fl):
        b, c, h, w = fg.size()
        fl = self.local_bn(self.local_conv(fl))

        fl_pos = self.posemb_local(fl)
        fl = fl + fl_pos

        fg_pos = self.posemb_global(fg)
        fg = fg_pos + fg

        fl_sample = F.interpolate(fl, size=(h, w), mode='bilinear', align_corners=True)
        
        fg = self.fusion_fg_fl_transformer(fg, fl_sample)
        
        return fg
        
        
class RetrievalNet(nn.Module):
    def __init__(self, backbone, pretrained=True, global_dim=2048, local_dim=128):
        super().__init__()
        self.backbone = ResNet(name=backbone, pretrained=pretrained)
        self.local_branch = StructBranch()
        
        # self.global_branch = FusionBranch()
        self.global_branch = MatchAlign()

        # Global Descriptor: pool --> whiten --> l2 norm
        self.pool = GeMPool(gem_p=cfg.GEM.P, trainable=cfg.GEM.TRAIN)
        # self.whiten = nn.Linear(global_dim, global_dim, bias=True)

    def forward(self, x):
        b = x.size(0)
        local_featmap, global_featmap = self.backbone(x)
        f_l = self.local_branch(local_featmap)
        f_g = self.global_branch(global_featmap, f_l)

        # global desc
        f_g = self.pool(f_g)
        f_g = f_g.view(b, -1)
        # f_g = self.whiten(f_g)
        # f_g = F.normalize(f_g, p=2, dim=1)

        return f_g