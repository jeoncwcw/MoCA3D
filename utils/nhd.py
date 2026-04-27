import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class NHDComputer:
    def __init__(self, device):
        self.device = device
        self.nhd_list = []
        self.total_num = 0
        self.f_v = self.h_v = 512.0 # Default virtual focal/height, matching IoU3DComputer

    def update(self, outputs, batch):
        '''
        outputs: dict containing:
            'corner coords': [B, 8, 2] (UV)
            'corner depths': [B, 8] or [B, 8, H, W]
        batch: dict containing:
            '3d_bbx': [B, 8, 3] (GT)
            'pad_left': [B,]
            'pad_top': [B,]
            'scale': [B,]
            'K': [B, 3, 3]
            'h': [B,] (Original Image Height)
        '''
        pred_uv = outputs['corner coords']
        pred_depths_raw = outputs['corner depths']
        gt_boxes = batch['3d_bbx']
        
        if isinstance(gt_boxes, list):
            gt_boxes = torch.stack([torch.as_tensor(x) for x in gt_boxes]).to(self.device).float()
            
        if pred_depths_raw.ndim == 4:
            pred_depths = self.sample_depths(pred_depths_raw, pred_uv / 4.0) # [B, 8]
        elif pred_depths_raw.ndim == 2:
            pred_depths = pred_depths_raw
        else:
            raise ValueError(f"Unexpected corner depths shape: {tuple(pred_depths_raw.shape)}")
        
        pred_uv_orig, pred_depths_orig = self.cal_original(
            pred_uv, pred_depths, 
            batch["pad_left"], batch["pad_top"], batch["scale"], 
            batch["K"], batch["h"]
        )
        
        # 3. Direct Unprojection to 3D using K
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        pred_corners_3d = self.unproject_corners(pred_uv_orig, pred_depths_orig, batch['K']) # [B, 8, 3]
        
        # 4. Compute NHD
        nhd_scores = self.compute_nhd_batch(pred_corners_3d, gt_boxes) # [B,]
        
        self.nhd_list.append(nhd_scores)
        self.total_num += pred_uv.shape[0]

    def return_result(self):
        if self.total_num == 0:
            return 0.0
            
        all_nhd = torch.cat(self.nhd_list, dim=0) # [N,]
        avg_nhd = all_nhd.mean().item()
        median_nhd = all_nhd.median().item()
        max_nhd = all_nhd.max().item()
        
        print(f"NHD Results over {self.total_num} samples:")
        print(f"  Avg NHD:    {avg_nhd:.4f}")
        print(f"  Median NHD: {median_nhd:.4f}")
        print(f"  Max NHD:    {max_nhd:.4f}")
        
        # Report success rates at various thresholds
        thresholds = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        print("\n  NHD Thresholds (% samples < thresh):")
        for thresh in thresholds:
            count = (all_nhd <= thresh).sum().item()
            ratio = (count / self.total_num) * 100
            print(f"    <= {thresh:4.2f}: {ratio:5.2f}%")
            
        return avg_nhd

    def unproject_corners(self, pred_uv, pred_depths, K):
        '''
        pred_uv: [B, 8, 2]
        pred_depths: [B, 8]
        K: [B, 3, 3]
        '''
        eps = 1e-6
        fx, fy, cx, cy = K[:, 0,0], K[:, 1,1], K[:, 0,2], K[:, 1,2] # [B,]
        
        u, v = pred_uv[..., 0], pred_uv[..., 1]
        z = pred_depths
        
        x = (u - cx[:, None]) * z / (fx[:, None] + eps)
        y = (v - cy[:, None]) * z / (fy[:, None] + eps)
        
        # Stack to form (x, y, z)
        P = torch.stack([x, y, z], dim=-1) # [B, 8, 3]
        return P

    def compute_nhd_batch(self, pred_corners, gt_corners):
        '''
        Compute NHD for a batch of predictions and GT.
        '''
        B = pred_corners.shape[0]
        nhd_scores = torch.zeros(B, device=pred_corners.device)
        
        # Ensure float
        pred_corners = pred_corners.float()
        gt_corners = gt_corners.float()

        for b in range(B):
            pred = pred_corners[b]
            gt = gt_corners[b]

            # Cost matrix
            cost_matrix = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0)).squeeze(0) # [8, 8]
            
            # Hungarian matching
            cost_np = cost_matrix.cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            
            # Sum of matched distances
            sum_dist = cost_matrix[row_ind, col_ind].sum()
            
            # Normalization (Max diagonal of GT)
            # Calculating max pairwise distance in GT corner set
            gt_pairwise = torch.cdist(gt.unsqueeze(0), gt.unsqueeze(0)).squeeze(0)
            d_gt = gt_pairwise.max()
            
            if d_gt < 1e-6:
                d_gt = 1.0
                
            nhd_scores[b] = (sum_dist / 8.0) / d_gt
            
        return nhd_scores

    def sample_depths(self, depth_maps, coords):
        """
        Copied from IoU3DComputer to ensure consistent sampling.
        depth_maps: [B, 8, H, W]
        coords: [B, 8, 2]
        """
        B, num_corners, H, W = depth_maps.shape
        
        # normalize coords to [-1, 1] for grid_sample
        norm_coords = coords.clone()
        norm_coords[..., 0] = 2.0 * (norm_coords[..., 0] / (W - 1)) - 1.0 # u
        norm_coords[..., 1] = 2.0 * (norm_coords[..., 1] / (H - 1)) - 1.0 # v
        
        flat_depth_maps = depth_maps.reshape(B * num_corners, 1, H, W)
        flat_grid = norm_coords.reshape(B * num_corners, 1, 1, 2)
        
        sample_depths = F.grid_sample(flat_depth_maps, flat_grid, mode="bilinear", align_corners=True)
        return sample_depths.reshape(B, num_corners)

    def cal_original(self, pred_uv, pred_depths, pad_left, pad_top, scale, K, h):
        if not isinstance(pad_left, torch.Tensor):
            pad_left = torch.tensor(pad_left, device=pred_uv.device, dtype=torch.float32)
        if not isinstance(pad_top, torch.Tensor):
            pad_top = torch.tensor(pad_top, device=pred_uv.device, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=pred_uv.device, dtype=torch.float32)
            
        pred_uv = pred_uv.clone()
        pred_uv[..., 0] = (pred_uv[..., 0] - pad_left.view(-1, 1)) / scale.view(-1, 1)
        pred_uv[..., 1] = (pred_uv[..., 1] - pad_top.view(-1, 1)) / scale.view(-1, 1)
        
        virtual_scale = ((h / K[..., 1, 1]) * (self.f_v / self.h_v)).unsqueeze(1)
        pred_depths = pred_depths / virtual_scale
        
        return pred_uv, pred_depths
