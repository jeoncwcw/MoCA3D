import torch
from torch import nn
import torch.nn.functional as F
from .heatmap import TargetGenerator

class BETRLoss(nn.Module):
    def __init__(self, cfg):
        super(BETRLoss, self).__init__()
        self.heatmap_size, self.input_size = int(cfg.heatmap_size), int(cfg.input_size)
        self.start_coarse, self.end_coarse = cfg.loss_weights.start_coarse, cfg.loss_weights.end_coarse
        self.start_depth, self.end_depth = cfg.loss_weights.start_depth, cfg.loss_weights.end_depth
        self.start_fine, self.end_fine = cfg.loss_weights.start_fine, cfg.loss_weights.end_fine
        self.max_depth = cfg.max_depth

        self.threshold = cfg.threshold
        self.peak = cfg.peak
        self.weight_depth = self.start_depth
        self.weight_coarse = self.start_coarse
        self.weight_fine = self.start_fine

        self.target_gen = TargetGenerator(heatmap_size=self.heatmap_size)
        self.total_epochs = cfg.num_epochs
        self.current_epoch = 0
        self.warmup_threshold = cfg.loss_weights.warmup_threshold
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def get_current_weights(self):
        progress = self.current_epoch / self.total_epochs
        loss_coarse = self.start_coarse + progress * (self.end_coarse - self.start_coarse)
        
        if progress < self.warmup_threshold:
            loss_depth = 0.0
            loss_fine = 0.0
        else:
            adj_progress = (progress - self.warmup_threshold) / (1.0 - self.warmup_threshold)
            loss_depth = self.start_depth + adj_progress * (self.end_depth - self.start_depth)
            loss_fine = self.start_fine + adj_progress * (self.end_fine - self.start_fine)
            
        return loss_coarse, loss_depth, loss_fine
    
    def forward(self, preds, batch):
        """
        preds['corner coords']: [B, 8, 2] (0-511 scale)
        preds['corner heatmaps']: [B, 8, 128, 128]
        preds['corner depths']: [B, 8]
        batch['gt_corners']: [B, 8, 2] (0-1 scale)
        batch['padding_mask']: [B, 512, 512] (bool, True for padding)
        """
        device = preds["corner heatmaps"].device
        raw_mask = batch["padding_mask"].unsqueeze(1).float()  # [B, 1, 512, 512]
        valid_mask = 1.0 - F.interpolate(raw_mask, size=(self.heatmap_size, self.heatmap_size), mode="nearest") # [B, 1, 128, 128]
        valid_mask = valid_mask.to(device)
        gt_corners = batch['gt_corners']  # [B, 8, 2]

        weight_map = self.target_gen.generate_heatmap(gt_corners, device) # [B, 8, 128, 128]
        weight_map = weight_map * valid_mask  # Mask out padding areas
       
        pred_confidence_map = preds['corner heatmaps'].detach()  # [B, 8, 128, 128]
        pred_confidence_map = pred_confidence_map * valid_mask  # Mask out padding areas
        
        self.weight_coarse, self.weight_depth, self.weight_fine = self.get_current_weights()
        # -- [Corner Loss] --
        # L_coarse
        B, C, _, _ = preds['corner heatmaps'].shape
        pos_weight = torch.where(weight_map > self.threshold, self.peak, 1.0) * valid_mask
        coarse_diff = F.smooth_l1_loss(preds['corner heatmaps'], weight_map, reduction='none', beta=0.1)
        loss_coarse = (coarse_diff * pos_weight).sum() / (pos_weight.sum() + 1e-8)
        # L_fine
        pred_corner_norm = preds['corner coords'] / float(self.input_size)
        loss_fine = F.smooth_l1_loss(pred_corner_norm, batch['gt_corners'], reduction='mean', beta=0.01)
        # total corner loss
        loss_corners = self.weight_coarse * loss_coarse + self.weight_fine * loss_fine

        # -- [Weighted Dense Losses] --
        gt_depth_sqrt = batch['gt_depths'].clamp(min=1e-4, max=self.max_depth) # [B, 8]
        gt_corner_depths_map = gt_depth_sqrt.view(-1, 8, 1, 1).expand(-1, 8, self.heatmap_size, self.heatmap_size)  # [B, 8, 128, 128]
        loss_depths = self._weighted_loss(preds['corner depths'], gt_corner_depths_map, pred_confidence_map)

        total_loss = loss_corners + self.weight_depth * loss_depths
        return {
            "total_loss": total_loss,
            "loss_corners": loss_corners,
            "loss_depths": loss_depths,
            "loss_details": {
                "loss_coarse": loss_coarse,
                "loss_fine": loss_fine,
            }
        }
        
    def _weighted_loss(self, pred, target, weight_map):
        loss = F.smooth_l1_loss(pred, target, reduction='none', beta=1.0)
        weighted_loss = (loss * weight_map).sum() / (weight_map.sum() + 1e-8)
        return weighted_loss
  