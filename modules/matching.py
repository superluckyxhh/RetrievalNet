import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformers import Attention, TransformerUnit, LinearTransformerUnit, LinearAttention
from modules.pool import GeMPool
from modules.position_embedding import PositionEmbSin2D

from core.config import cfg


def compute_match(feat0, feat1):
    # [b, c, h, w] ---> [b, n, c]
    b, c, h, w = feat0.size()
    feat0 = feat0.flatten(start_dim=2).transpose(1, 2)
    feat0_norm = F.normalize(feat0, dim=1) 
    
    # [b, c, h, w] ---> [b, m, c]
    feat1 = feat1.flatten(start_dim=2).transpose(1, 2)
    feat1_norm = F.normalize(feat1, dim=1)
    
    # Compute Similarity
    scores = torch.einsum("bnc,bmc->bnm", feat0_norm, feat1_norm)
    # Compute Prob
    prob = F.softmax(scores, dim=-1)
    # Compute Value
    value = torch.bmm(prob, feat0_norm).view(b, c, h, w)

    return value, prob, scores


def compute_match_scores(scores, temperature=0.1, eps=1e-10):
    scores = scores / temperature
    scores_col = F.softmax(scores, dim=1)
    scores_row = F.softmax(scores, dim=2)
    scores = -1 * torch.log(scores_col * scores_row + eps)
    
    return scores
    
    
class MatchAlign(nn.Module):
    def __init__(self, global_in_planes=2048, local_in_planes=1024):
        super().__init__()
        self.div_dim = global_in_planes ** -0.5
        
        self.conv = nn.Conv2d(local_in_planes, global_in_planes, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(global_in_planes)

        # self.transformer = TransformerUnit(global_in_planes, Attention(global_in_planes), mlp_ratio=1)
        self.transformer = LinearTransformerUnit(global_in_planes, LinearAttention(global_in_planes), mlp_ratio=1)
        self.pos_embed_local = PositionEmbSin2D(global_in_planes)
        self.pos_embed_global = PositionEmbSin2D(global_in_planes)
        
        self.merge = nn.Conv2d(global_in_planes, global_in_planes, kernel_size=1)
        self.merge_bn = nn.BatchNorm2d(global_in_planes)
    
    def forward(self, fg, fl):
        b, cg, hg, wg = fg.size()
        _, cl, bl, hl = fl.size()
        
        # Expanded dim of local featmap
        fl = F.interpolate(fl, size=(hg, wg), mode='bilinear', align_corners=True)
        fl = self.conv(fl)
        fl = self.bn(fl)
        
        # Add position embedding
        fl_pos = self.pos_embed_local(fl)
        fl = fl + fl_pos
        fg_pos = self.pos_embed_global(fg)
        fg = fg + fg_pos

        # Compute corss attention
        fg = self.transformer(fg, fl)

        # Compute feature similarity and probability
        align, match_prob, match_scores = compute_match(fg, fl)

        fg = self.merge(fg + align)
        fg = self.merge_bn(fg)
        
        return fg