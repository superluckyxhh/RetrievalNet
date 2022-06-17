import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionEmbSin2D(nn.Module):
    def __init__(self, d_model, max_shape=(256, 256), temperature=10000.):
        super().__init__()
        dim = d_model // 2

        pe = torch.zeros((d_model, *max_shape))
        pe.requires_grad = False

        y_pos = torch.ones(max_shape).cumsum(0).float().unsqueeze(0) # [1, 256, 256]
        x_pos = torch.ones(max_shape).cumsum(1).float().unsqueeze(0) # [1, 256, 256]

        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(temperature)/dim)) #[64]
        div_term = div_term[:, None, None] #[64, 1, 1]
        # x0, y0, x1, y1 ->4
        pe[0::4, :, :] = torch.sin(x_pos * div_term)
        pe[1::4, :, :] = torch.cos(x_pos * div_term)
        pe[2::4, :, :] = torch.sin(y_pos * div_term)
        pe[3::4, :, :] = torch.cos(y_pos * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor):
        return self.pe[:, :, :x.size(2), :x.size(3)] 


class PositionEmbSin1D(nn.Module):
    def __init__(
        self,
        d_model: int, 
        max_len: int=2048, 
        temperature: float = 10000.,
    ):
        super().__init__()
        dim = d_model // 2

        pe = torch.zeros(max_len, d_model).float().unsqueeze(0)
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(temperature) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return self.pe[:, x.size(2)]
