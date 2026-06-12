import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

class BBox3DMLP(nn.Module):
    '''
    A simple MLP to predict 3D bounding box parameters from the Image plane geometry (output of MoCA3D)
    '''
    def __init__(self, hidden_dim=128):
        super(BBox3DMLP, self).__init__()
        self.roi_out_size = 7
        self.roi_feat_dim = 128
        self.roi_erase_prob = 0.3
        self.roi_erase_min_ratio = 0.15
        self.roi_erase_max_ratio = 0.45
        input_dim = (8 * 3) + 5 + self.roi_feat_dim
        # MoCA default decoder channel is d_model=384.
        self.roi_reduce = nn.Sequential(
            nn.Conv2d(384, self.roi_feat_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.roi_norm = nn.LayerNorm(self.roi_feat_dim)
        self.roi_dropout = nn.Dropout(p=0.1)
        self.feat_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 13),
        )

    def _apply_roi_erasing(self, roi_feat):
        # Randomly erase one rectangle in each sampled ROI map (train-time only).
        if (not self.training) or self.roi_erase_prob <= 0.0:
            return roi_feat
        B, _, H, W = roi_feat.shape
        if H < 2 or W < 2:
            return roi_feat

        apply_mask = torch.rand(B, device=roi_feat.device) < self.roi_erase_prob
        if not apply_mask.any():
            return roi_feat

        out = roi_feat.clone()
        spatial_mask = torch.ones((B, 1, H, W), device=roi_feat.device, dtype=roi_feat.dtype)
        min_h = max(1, int(H * self.roi_erase_min_ratio))
        max_h = max(min_h + 1, int(H * self.roi_erase_max_ratio))
        min_w = max(1, int(W * self.roi_erase_min_ratio))
        max_w = max(min_w + 1, int(W * self.roi_erase_max_ratio))

        for b in range(B):
            if not apply_mask[b]:
                continue
            eh = int(torch.randint(min_h, max_h + 1, (1,), device=roi_feat.device).item())
            ew = int(torch.randint(min_w, max_w + 1, (1,), device=roi_feat.device).item())
            eh = min(eh, H)
            ew = min(ew, W)
            y1 = int(torch.randint(0, H - eh + 1, (1,), device=roi_feat.device).item())
            x1 = int(torch.randint(0, W - ew + 1, (1,), device=roi_feat.device).item())
            spatial_mask[b, :, y1:y1 + eh, x1:x1 + ew] = 0.0

        out = out * spatial_mask
        keep_ratio = spatial_mask.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-3)
        out = out / keep_ratio
        return out

    def forward(self, x, K):
        '''
        Args:
            x: Dict with keys
                - 'corner coords': [B, 8, 2] - projected uv
                - 'sampled depths': [B, 8] - depth values at the 8 corners
        Returns:
            Dict with center/size/yaw/uncertainty and ray direction terms.
        '''
        corner_coords = x['corner coords']  # [B, 8, 2]
        sampled_depths = x['sampled depths']  # [B, 8]

        fx = K[:, 0, 0]  # [B]
        fy = K[:, 1, 1]  # [B]
        cx = K[:, 0, 2]  # [B]
        cy = K[:, 1, 2]  # [B]
        eps = 1e-6

        u = corner_coords[..., 0]  # [B, 8]
        v = corner_coords[..., 1]  # [B, 8]
        z = torch.clamp(sampled_depths, min=eps)  # [B, 8]

        x_norm = (u - cx.unsqueeze(1)) / (fx.unsqueeze(1) + eps)  # [B, 8]
        y_norm = (v - cy.unsqueeze(1)) / (fy.unsqueeze(1) + eps)  # [B, 8]

        # Prior in normalized image plane + depth.
        x0 = x_norm.mean(dim=1)  # [B]
        y0 = y_norm.mean(dim=1)  # [B]
        z0 = z.mean(dim=1)  # [B]

        # Physically consistent 3D center prior from per-corner backprojection.
        X_i = x_norm * z  # [B, 8]
        Y_i = y_norm * z  # [B, 8]
        X0 = X_i.mean(dim=1)  # [B]
        Y0 = Y_i.mean(dim=1)  # [B]

        norm_u = x_norm - x0.unsqueeze(1)  # [B, 8]
        norm_v = y_norm - y0.unsqueeze(1)  # [B, 8]
        norm_z = torch.log(z) - torch.log(z0.unsqueeze(1) + eps)  # [B, 8]
        local_shape_feats = torch.stack([norm_u, norm_v, norm_z], dim=-1).view(norm_u.size(0), -1)  # [B, 24]

        global_context_feats = torch.stack([
            x0,
            y0,
            z0,
            X0,
            Y0,
        ], dim=-1)  # [B, 5]
        feats = torch.cat([local_shape_feats, global_context_feats], dim=-1)  # [B, 29]

        decoder_feat = x.get('decoder_feat', x.get('decoder feat', None))
        bbx2d_tight = x.get('2d_bbx', None)
        if decoder_feat is not None and bbx2d_tight is not None:
            B = decoder_feat.shape[0]
            # Fixed geometry: input image 512x512 and decoder feature 32x32 -> stride 16.
            image_size = 512.0
            feat_stride = 16.0
            boxes = bbx2d_tight.to(device=decoder_feat.device, dtype=torch.float32).clone()
            boxes[:, [0, 2]] *= image_size
            boxes[:, [1, 3]] *= image_size
            boxes[:, 0].clamp_(0.0, image_size - 1e-3)
            boxes[:, 1].clamp_(0.0, image_size - 1e-3)
            boxes[:, 2].clamp_(1e-3, image_size)
            boxes[:, 3].clamp_(1e-3, image_size)
            boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1e-3)
            boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1e-3)

            # Keep RoIs in fp32 for numerical stability under AMP.
            batch_idx = torch.arange(B, device=decoder_feat.device, dtype=torch.float32).unsqueeze(1)
            rois = torch.cat([batch_idx, boxes], dim=1)
            roi_feat = roi_align(
                decoder_feat,
                rois,
                output_size=(self.roi_out_size, self.roi_out_size),
                spatial_scale=1.0 / feat_stride,
                sampling_ratio=2,
                aligned=True,
            )
            roi_feat = self._apply_roi_erasing(roi_feat)
            roi_feat = self.roi_reduce(roi_feat).flatten(1)  # [B, roi_feat_dim]
        else:
            roi_feat = feats.new_zeros((feats.size(0), self.roi_feat_dim))
        roi_feat = self.roi_norm(roi_feat)
        roi_feat = self.roi_dropout(roi_feat)
        feats = torch.cat([feats, roi_feat], dim=-1)
        feats = self.feat_norm(feats)
        
        bbox_3d_params = self.mlp(feats)  # [B, 13]
        dxn, dyn, dlogz = bbox_3d_params[:, 0], bbox_3d_params[:, 1], bbox_3d_params[:, 2]  # [B]

        # Cube R-CNN style center parameterization: residuals in (x_norm, y_norm, log z).
        x_hat = x0 + dxn
        y_hat = y0 + dyn
        z_hat = z0 * torch.exp(torch.clamp(dlogz, min=-6.0, max=6.0))

        centers = torch.stack([x_hat * z_hat, y_hat * z_hat, z_hat], dim=-1)  # [B, 3]
        center_prior = torch.stack([X0, Y0, z0], dim=-1)  # [B, 3]
        center_delta = centers - center_prior  # [B, 3], keeps legacy XYZ residual semantics
        center_delta_param = torch.stack([dxn, dyn, dlogz], dim=-1)  # [B, 3]
        # Cube-like size prior from mean center-to-corner radius in 3D:
        # r = mean(||corner - center_prior||), s0 = 2 / sqrt(3) * r.
        corner_3d = torch.stack([X_i, Y_i, z], dim=-1)  # [B, 8, 3]
        center_prior = torch.stack([X0, Y0, z0], dim=-1)  # [B, 3]
        corner_radius = torch.norm(corner_3d - center_prior.unsqueeze(1), dim=-1).mean(dim=1)  # [B]
        size0 = (2.0 / (3.0 ** 0.5)) * corner_radius
        size_prior = size0.unsqueeze(-1).expand(-1, 3).clamp_min(1e-3)  # [B, 3]
        dlog_size = torch.clamp(bbox_3d_params[:, 3:6], min=-6.0, max=6.0)  # [B, 3]
        sizes = (size_prior * torch.exp(dlog_size)).clamp_min(1e-3)  # [B, 3]
        yaws = bbox_3d_params[:, 6:12]  # [B, 6]
        uncertainty = bbox_3d_params[:, 12:]  # [B, 1]
        uncertainty = F.softplus(uncertainty) + 1e-3  # [B, 1]
        return {
            'centers': centers,
            'center_prior': center_prior,
            'center_delta': center_delta,
            'center_prior_param': torch.stack([x0, y0, z0], dim=-1),
            'center_delta_param': center_delta_param,
            'sizes': sizes,
            'size_prior': size_prior,
            'size_delta': sizes - size_prior,
            'size_delta_param': dlog_size,
            'yaws': yaws,
            'uncertainty': uncertainty,
            'ray_z': z0,
            'ray_x': X0,
            'ray_y': Y0,
        }
        
