import torch
import numpy as np

class TargetGenerator:
    def __init__(self, heatmap_size=128):
        self.heatmap_size = heatmap_size
        y, x = torch.meshgrid(
            torch.arange(heatmap_size),
            torch.arange(heatmap_size),
            indexing='ij'
        )
        self.grid_y = y.float()
        self.grid_x = x.float()

    def generate_heatmap(self, gt_corners, device):
        """       
        :param gt_corners: [B, 8, 2] Normalized coordinates of corners (0-1)
        :param device: torch device
        Returns: [B, 8, 128, 128] weight map
        """
        B = gt_corners.shape[0]
        H = W = self.heatmap_size
        
        max_norm = (self.heatmap_size - 1) / self.heatmap_size
        gt_px = gt_corners.clamp(min=0.0, max=max_norm) * self.heatmap_size
        gt_x = gt_px[:, :, 0].view(B, 8, 1, 1)
        gt_y = gt_px[:, :, 1].view(B, 8, 1, 1)
        gt_center = gt_px.mean(dim=1, keepdim=True)  # [B, 1, 2]
        
        dist_to_center = torch.norm(gt_px - gt_center, dim=2)
        denom = (0.2 * dist_to_center).pow(2).clamp(min=2.0)
        denom = denom.view(B, 8, 1, 1)
        grid_x = self.grid_x.to(device).view(1, 1, H, W)
        grid_y = self.grid_y.to(device).view(1, 1, H, W)
        
        dist_sq = (grid_x - gt_x).pow(2) + (grid_y - gt_y).pow(2)
        weight_map = torch.exp(-dist_sq / (denom))
        return weight_map # [B, 8, 128, 128]
