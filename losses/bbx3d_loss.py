import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

class BBox3DLoss(nn.Module):
    def __init__(self):
        super(BBox3DLoss, self).__init__()
    
    def chamfer_dist(self, pred_boxes, gt_boxes):
        dist = torch.cdist(pred_boxes, gt_boxes)  # [B, N_pred, N_gt]
        min_dist_pred = dist.min(dim=2)[0].mean(dim=1)
        min_dist_gt = dist.min(dim=1)[0].mean(dim=1)
        return min_dist_pred + min_dist_gt

    def l1_distance(self, pred_boxes, gt_boxes):
        return torch.abs(pred_boxes - gt_boxes).mean(dim=(1, 2))
    
    def build_bbox3d(self, centers, sizes, allocentric_6d, ray_x, ray_z, ray_y=None):
        B = centers.shape[0]
        device = centers.device
        
        R_allo = rotation_6d_to_matrix(allocentric_6d)
        
        theta_ray = torch.atan2(ray_x, ray_z)
        cos_t, sin_t = torch.cos(theta_ray), torch.sin(theta_ray)
        zeros, ones = torch.zeros_like(theta_ray), torch.ones_like(theta_ray)
        
        R_ray = torch.stack([
            torch.stack([cos_t, zeros, sin_t], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin_t, zeros, cos_t], dim=-1)
        ], dim=-2) # [B, 3, 3]
        R_ego = torch.bmm(R_ray, R_allo)  # [B, 3, 3]
        
        x_c = torch.tensor([-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5], device=device)
        y_c = torch.tensor([-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5], device=device)
        z_c = torch.tensor([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5], device=device)
        
        unit_box = torch.stack([x_c, y_c, z_c], dim=-1).unsqueeze(0).expand(B, -1, -1)  # [B, 8, 3]
        scaled_box = unit_box * sizes.unsqueeze(1)  # [B, 8, 3]
        rotated_box = torch.bmm(scaled_box, R_ego.transpose(1, 2))  # [B, 8, 3]
        final_box = rotated_box + centers.unsqueeze(1)  # [B, 8, 3]
        return final_box
    
    def forward(self, preds, gts):
        ray_x = preds['ray_x']
        ray_y = preds.get('ray_y', None)
        ray_z = preds['ray_z']
        gt_corners = gts['corners']
        
        # -- 1. L_all_3D --
        preds_boxes_all = self.build_bbox3d(preds['centers'], preds['sizes'], preds['yaws'], ray_x, ray_z, ray_y=ray_y)
        loss_all = self.chamfer_dist(preds_boxes_all, gt_corners)
        
        # -- 2. L_center_3D --
        box_center_only = self.build_bbox3d(preds['centers'], gts['sizes'], gts['yaws'], ray_x, ray_z, ray_y=ray_y)
        loss_center = self.l1_distance(box_center_only, gt_corners)
        
        # -- 3. L_size_3D --
        box_size_only = self.build_bbox3d(gts['centers'], preds['sizes'], gts['yaws'], ray_x, ray_z, ray_y=ray_y)
        loss_size = self.l1_distance(box_size_only, gt_corners)
        
        # -- 4. L_yaw_3D --
        box_yaw_only = self.build_bbox3d(gts['centers'], gts['sizes'], preds['yaws'], ray_x, ray_z, ray_y=ray_y)
        loss_yaw = self.l1_distance(box_yaw_only, gt_corners)
        
        # -- 5. Total Loss --
        L_3D = loss_center + loss_size + loss_yaw + loss_all
        mu = preds['uncertainty'].squeeze(-1)  # [B]
        
        total_loss = 1.414 * torch.exp(-mu) * L_3D + mu
        
        loss_dict = {
            'loss_all': loss_all.mean(),
            'loss_center': loss_center.mean(),
            'loss_size': loss_size.mean(),
            'loss_yaw': loss_yaw.mean(),
            'total_loss': total_loss.mean(),
        }
        
        return loss_dict

def prepare_gt_for_loss(gt_3d_bbx):
    centers = gt_3d_bbx.mean(dim=1) # [B, 3]
    
    # Sizes
    v0 = gt_3d_bbx[:, 0, :]
    v1 = gt_3d_bbx[:, 1, :]
    v3 = gt_3d_bbx[:, 3, :]
    v4 = gt_3d_bbx[:, 4, :]
    
    width = torch.norm(v1 - v0, dim=-1)   # X축
    height = torch.norm(v3 - v0, dim=-1)  # Y축
    length = torch.norm(v4 - v0, dim=-1)  # Z축
    sizes = torch.stack([width, height, length], dim=-1) # [B, 3]
    
    # Egocentric 3x3 Rotation Matrix (R_ego)
    x_axis = F.normalize(v1 - v0, dim=-1)
    y_axis = F.normalize(v3 - v0, dim=-1)
    z_axis = F.normalize(v4 - v0, dim=-1)
    
    R_ego = torch.stack([x_axis, y_axis, z_axis], dim=-1) # [B, 3, 3]
    
    # Ray Angle
    ray_x = centers[:, 0]
    ray_y = centers[:, 1]
    ray_z = centers[:, 2]
    theta_ray = torch.atan2(ray_x, ray_z) # [B]
    
    cos_t, sin_t = torch.cos(theta_ray), torch.sin(theta_ray)
    zeros, ones = torch.zeros_like(theta_ray), torch.ones_like(theta_ray)
    
    R_ray = torch.stack([
        torch.stack([cos_t, zeros, sin_t], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sin_t, zeros, cos_t], dim=-1)
    ], dim=-2) # [B, 3, 3]
    
    # Egocentric -> Allocentric Transformation
    R_allo = torch.bmm(R_ray.transpose(1, 2), R_ego)
    
    # Allocentric 3x3 -> 6D Continuous Transformation
    allocentric_6d = matrix_to_rotation_6d(R_allo) # [B, 6]
    
    # Final Return
    gts = {
        'corners': gt_3d_bbx,      # [B, 8, 3] Loss의 기준점!
        'centers': centers,        # [B, 3]
        'sizes': sizes,            # [B, 3]
        'yaws': allocentric_6d,    # [B, 6]
        'ray_x': ray_x,            # [B]
        'ray_y': ray_y,            # [B]
        'ray_z': ray_z,            # [B]
    }
    
    return gts
