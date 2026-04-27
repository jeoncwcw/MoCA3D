import torch
from torch import nn
import torch.nn.functional as F

class DirectLoss(nn.Module):
    def __init__(self, cfg):
        super(DirectLoss, self).__init__()
        self.heatmap_size, self.input_size = int(cfg.heatmap_size), int(cfg.input_size)
        self.max_depth = cfg.max_depth
        self.weight_depth = cfg.loss_weights.depth
        self.weight_fine = cfg.loss_weights.fine
        
    def forward(self, preds, batch):
        """
        preds['corner coords']: [B, 8, 2]
        preds['corner depths']: [B, 8]
        batch['gt_corners']: [B, 8, 2] (0-1 scale)
        """
        # -- [Corner Loss] --
        pred_normalized_coords = preds['corner coords'] / self.input_size  # Normalize to [0, 1]
        loss_fine = F.smooth_l1_loss(pred_normalized_coords, batch['gt_corners'], reduction='mean', beta=0.01)
        # -- [Depth Loss] --
        normalized_gt_depths = (batch['gt_depths'] / self.max_depth).clamp(0, 1)  # Normalize to [0, 1]
        loss_depths = F.smooth_l1_loss(preds['corner depths'], normalized_gt_depths, reduction='mean', beta=0.01)

        # -- [Weighted Dense Losses] --
        total_loss = self.weight_fine * loss_fine + self.weight_depth * loss_depths
        return {
            "total_loss": total_loss,
            "loss_corners": loss_fine,
            "loss_depths": loss_depths
        }
        
  