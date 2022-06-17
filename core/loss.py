import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from core.config import cfg
from modules.pool import GeMPool


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class CrossEntropyProduct(nn.Module):
    def __init__(self, num_classes, global_dim=2048):
        super().__init__()
        self.fc = nn.Linear(global_dim, num_classes, bias=True)
        # self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        return self.fc(x)


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, num_classes=81313, weight=None, 
                 reduction="mean", class_weights_norm=None):
        super().__init__()
        self.arcface_product = ArcMarginProduct(2048, num_classes)
        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm
        self.crit = nn.CrossEntropyLoss(reduction="none")   
        
        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, feature, labels):
        logits = self.arcface_product(feature)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)
        s = self.s
        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)
            loss = loss * w
            if self.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()
            
            return {'loss':loss}


        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return {'loss':loss}


class CrossEntropy(nn.Module):
    def __init__(self, num_classes=81313):
        super().__init__()
        self.cre_product = CrossEntropyProduct(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, feature, targets):
        logits = self.cre_product(feature)
        loss = self.criterion(logits, targets)

        return {"loss": loss}


class LocalLoss(nn.Module):
    def __init__(self, local_dim=128, num_classes=81313):
        super().__init__()
        self.pool = GeMPool()
        self.fc = nn.Linear(local_dim, num_classes, bias=True)
        self.criterion = nn.CrossEntropyLoss()
        self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, feature, targets):
        feature = self.pool(feature).squeeze(-1).squeeze(-1)
        logits = self.fc(feature)
        loss = self.criterion(logits, targets)

        return {"loss": loss}
    

class Criterion(nn.Module):
    def __init__(self, s, m, num_classes=81313):
        super().__init__()
        
        if cfg.LOSS.GLOBAL_NAME == 'arcface':  
            self.global_criterion = ArcFaceLoss(s, m, num_classes)
        elif cfg.LOSS.GLOBAL_NAME == 'cross_entropy':
            self.global_criterion = CrossEntropy(num_classes)
        else:
            raise ValueError("Invalid Global Loss Name")
      
        if cfg.LOSS.LOCAL_NAME == 'cross_entropy':
            self.cross_entropy = LocalLoss(128, num_classes)
        else:
            raise ValueError('Invalid Local Loss Name')
        
    def forward(self, global_feats, local_feats, labels):
        global_loss = self.global_criterion(global_feats, labels) * cfg.LOSS.GLOBAL_WEIGHT
        local_loss = self.cross_entropy(local_feats, labels) * cfg.LOSS.LOCAL_WEIGHT
        total_loss = global_loss['loss'] + local_loss['loss']
        
        return {'global_loss': global_loss['loss'], 
                'local_loss': local_loss['loss'],
                'loss': total_loss
        }


def setup_criterion():
    return Criterion(s=cfg.LOSS.ARCFACE_SCALE, 
                    m=cfg.LOSS.ARCFACE_MARGIN, 
                    num_classes=cfg.TRAIN.DATASET_NUM_CLASS
    )


def set_local_criterion():
    return LocalLoss(local_dim=1024, num_classes=cfg.TRAIN.DATASET_NUM_CLASS)

def setup_global_criterion():
    if cfg.LOSS.GLOBAL_NAME == 'arcface':  
        return ArcFaceLoss(
                s=cfg.LOSS.ARCFACE_SCALE,
                m=cfg.LOSS.ARCFACE_MARGIN,
                num_classes=cfg.TRAIN.DATASET_NUM_CLASS
        )
    elif cfg.LOSS.GLOBAL_NAME == 'cross_entropy':
        return CrossEntropy(num_classes=cfg.TRAIN.DATASET_NUM_CLASS)
    else:
        raise ValueError("Invalid Global Loss Name")


def topk_errors(preds, labels, ks):
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks]
    
    return [(1.0 - x / preds.size(0)) * 100.0 for x in topks_correct]