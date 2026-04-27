import torch
from torch import nn
import torch.nn.functional as F

class SoftArgmax2D(nn.Module):
    def __init__(self, beta=100.0, is_sigmoid=False):
        super(SoftArgmax2D, self).__init__()
        self.beta = beta
        self.is_sigmoid = is_sigmoid

    def forward(self, x, valid_mask=None):
        B, C, H, W = x.shape
        if self.is_sigmoid:
            x = torch.sigmoid(x)
            
        x = x.view(B, C, -1)
        x = F.softmax(self.beta * x, dim=-1).view(B, C, H, W)

        if valid_mask is not None:
            if valid_mask.shape[1] == 1 and C != 1:
                valid_mask = valid_mask.expand(B, C, H, W)
            x = x * valid_mask  # Mask out invalid positions
            denom = x.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)  # Avoid division by zero
            x = x / denom
        
        device = x.device
        
        ticks_x = torch.arange(W, dtype = torch.float32, device=device)
        ticks_y = torch.arange(H, dtype = torch.float32, device=device)

        prob_x = x.sum(dim=2)  # (B, C, W)
        prob_y = x.sum(dim=3)  # (B, C, H)

        coords_x = (prob_x * ticks_x).sum(dim=-1, keepdim=True)  # (B, C, 1)
        coords_y = (prob_y * ticks_y).sum(dim=-1, keepdim=True)  # (B, C, 1)

        coords = torch.cat([coords_x, coords_y], dim=-1)  # (B, C, 2)

        return coords
