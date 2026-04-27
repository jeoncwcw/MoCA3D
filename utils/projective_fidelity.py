import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class CornerGeometryMetric:
    def __init__(self, device="cuda", use_hungarian_matching=True, direct_mode=False):
        self.device = device
        self.use_hungarian_matching = use_hungarian_matching
        self.direct_mode = direct_mode
        self.reset()
        self.f_v = self.h_v = 512.0
        self.max_depth = 100.0
        
    def reset(self):
        self.total_uv_dist = torch.tensor(0.0, device=self.device)
        self.total_uv_dist_naive = torch.tensor(0.0, device=self.device)  # For comparison
        self.total_depth_diff = torch.tensor(0.0, device=self.device)
        self.total_samples = torch.tensor(0, device=self.device, dtype=torch.long)
        self.total_depth_diff_rate = torch.tensor(0, device=self.device, dtype=torch.float32)
    # def sample_depths(self, depth_maps, coords):
    #     """
    #     depth_maps: [B, 8, H, W]
    #     coords: [B, 8, 2] - (u,v) coordinates
    #     """
    #     B, num_corners, H, W = depth_maps.shape
        
    #     # normalize coords to [-1, 1] for grid_sample
    #     norm_coords = coords.clone()
    #     norm_coords[..., 0] = 2.0 * (norm_coords[..., 0] / (W - 1)) - 1.0 # u
    #     norm_coords[..., 1] = 2.0 * (norm_coords[..., 1] / (H - 1)) - 1.0 # v
        
    #     flat_depth_maps = depth_maps.reshape(B * num_corners, 1, H, W)
    #     flat_grid = norm_coords.view(B * num_corners, 1, 1, 2)
        
    #     sample_depths = F.grid_sample(flat_depth_maps, flat_grid, mode="bilinear", align_corners=True)
    #     return sample_depths.view(B, num_corners) # [B, 8]
    
    def hungarian_match_corners(self, pred_uv, gt_uv):
        """
        Find optimal 1:1 matching between pred and GT corners using Hungarian algorithm.
        
        pred_uv: [B, 8, 2]
        gt_uv: [B, 8, 2]
        
        Returns:
            matched_pred_uv: [B, 8, 2] - reordered pred_uv based on optimal matching
            matched_indices: [B, 8] - indices showing which pred corner matched which GT corner
        """
        B = pred_uv.shape[0]
        matched_pred_uv = torch.zeros_like(pred_uv)
        matched_indices = torch.zeros(B, 8, dtype=torch.long, device=pred_uv.device)
        
        for b in range(B):
            # Compute pairwise distance matrix [8, 8]
            pred_b = pred_uv[b]  # [8, 2]
            gt_b = gt_uv[b]      # [8, 2]
            
            # cost[i, j] = distance from pred corner i to gt corner j
            cost_matrix = torch.cdist(pred_b.unsqueeze(0), gt_b.unsqueeze(0)).squeeze(0)  # [8, 8]
            
            # Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            
            # Reorder pred corners to match GT order
            for gt_idx, pred_idx in zip(col_ind, row_ind):
                matched_pred_uv[b, gt_idx] = pred_uv[b, pred_idx]
                matched_indices[b, gt_idx] = pred_idx
        
        return matched_pred_uv, matched_indices
    
    def update(self, outputs, batch):
        """
        outputs: model prediction dictionary
        batch: GT batch
        """
        # Get predictions and GT - clone to avoid CUDA Graphs issues
        pred_uv = outputs['corner coords'].detach().clone() # [B, 8, 2]
        if self.direct_mode:
            pred_depths = outputs['corner depths'].detach().clone() # [B, 8]
            pred_depths = pred_depths*self.max_depth  # Scale back to metric depth
        else:
            pred_depths = outputs['sampled depths'].detach().clone() # [B, 8]
        gt_uv = batch['gt_corners'].clone() * 512.0 # [B, 8, 2] - scale to 0-511
        gt_d = batch['gt_depths'].clone() # [B, 8] - in meters
        H_real = batch['h']
        f_real = batch["K"][..., 1, 1]
        
        # Ensure H_real is a tensor on the same device
        if not isinstance(H_real, torch.Tensor):
            H_real = torch.tensor(H_real, device=self.device, dtype=torch.float32)
        if not isinstance(f_real, torch.Tensor):
            f_real = torch.tensor(f_real, device=self.device, dtype=torch.float32)
        
        # Calculate naive (index-dependent) UV error for comparison
        naive_uv_dist = torch.norm(pred_uv - gt_uv, p=2, dim=-1)  # [B, 8]
        self.total_uv_dist_naive += naive_uv_dist.sum()
        
        # Hungarian matching for index-invariant comparison
        if self.use_hungarian_matching:
            matched_pred_uv, matched_indices = self.hungarian_match_corners(pred_uv, gt_uv)
            uv_dist = torch.norm(matched_pred_uv - gt_uv, p=2, dim=-1)  # [B, 8]
        else:
            uv_dist = naive_uv_dist
        pred_d = pred_depths
        
        # if pred_depths.ndim == 4:
        #     pred_uv_128 = pred_uv / 4.0
        #     pred_d = self.sample_depths(pred_depths, pred_uv_128)
        # elif pred_depths.ndim == 2:
        #     pred_d = pred_depths
        # else:
        #     raise ValueError(f"Unexpected corner depths shape: {tuple(pred_depths.shape)}")

        virtual_scale = (H_real / f_real) * (self.f_v / self.h_v)
        virtual_scale = virtual_scale.unsqueeze(1)
        pred_d = pred_d / virtual_scale
        gt_d = gt_d / virtual_scale
        
        # Depth error (using matched indices if Hungarian matching enabled)
        if self.use_hungarian_matching:
            # Reorder pred_d to match GT
            B = pred_d.shape[0]
            matched_pred_d = torch.zeros_like(pred_d)
            for b in range(B):
                for gt_idx in range(8):
                    pred_idx = matched_indices[b, gt_idx].item()
                    matched_pred_d[b, gt_idx] = pred_d[b, pred_idx]
            depth_diff = torch.abs(matched_pred_d - gt_d)
            depth_diff_rate = torch.abs(matched_pred_d - gt_d) / gt_d
        else:
            depth_diff = torch.abs(pred_d - gt_d)
            depth_diff_rate = depth_diff / gt_d.clamp_min(1e-8)
        
        self.total_uv_dist += uv_dist.sum()
        self.total_depth_diff += depth_diff.sum()
        self.total_samples += pred_uv.shape[0]
        self.total_depth_diff_rate += depth_diff_rate.sum()
        
    def compute(self):
        if self.total_samples == 0:
            return 0.0, 0.0, 0.0
        total_corners = self.total_samples * 8
        avg_uv_error = (self.total_uv_dist / total_corners).item()
        avg_depth_error = (self.total_depth_diff / total_corners).item()
        avg_depth_diff_rate = (self.total_depth_diff_rate / total_corners).item()
        return avg_uv_error, avg_depth_error, avg_depth_diff_rate, avg_uv_error + avg_depth_error * 5.0
    
    def get_result_dict(self):
        if self.total_samples == 0:
            return {
                "avg_uv_error": 0.0,
                "avg_uv_error_naive": 0.0,
                "avg_depth_error": 0.0,
                "avg_depth_diff_rate": 0.0,
                "combined_error": 0.0
            }
        total_corners = self.total_samples * 8
        avg_uv_error = (self.total_uv_dist / total_corners).item()
        avg_uv_error_naive = (self.total_uv_dist_naive / total_corners).item()
        avg_depth_error = (self.total_depth_diff / total_corners).item()
        avg_depth_diff_rate = (self.total_depth_diff_rate / total_corners).item()
        combined_error = avg_uv_error + avg_depth_error * 5.0
        return {
            "avg_uv_error": avg_uv_error,
            "avg_uv_error_naive": avg_uv_error_naive,  # For comparison: index-dependent error
            "avg_depth_error": avg_depth_error,
            "avg_depth_diff_rate": avg_depth_diff_rate,
            "combined_error": combined_error,
        }
