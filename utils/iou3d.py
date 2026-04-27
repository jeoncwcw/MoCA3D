import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
from pytorch3d import _C
from pytorch3d.ops.iou_box3d import _box_planes, _box_triangles

class IoU3DComputer:
    def __init__(
        self,
        device,
        rectify_mode: str = "pca",
        use_corner_confidence: bool = False,
        infer_order_with_k: bool = True,
        direct_mode: bool = False,
        max_depth: float = 100.0,
    ):
        self.iou_list = []
        self.iou_list_convex = []  # For invariant IoU (symmetry aware)
        self.iou_list_hungarian = [] # For hungarian IoU (fully shuffling robust)
        self.total_num = 0
        self.device = device
        self.f_v = self.h_v = 512.0
        self.rectify_mode = rectify_mode.lower()
        self.use_corner_confidence = bool(use_corner_confidence)
        self.infer_order_with_k = bool(infer_order_with_k)
        self.direct_mode = bool(direct_mode)
        self.max_depth = float(max_depth)
    
    def update(self, outputs, batch):
        '''
        outputs: dict containing:
            'corner coords': [B, 8, 2]
            - direct_mode=True:  'corner depths' [B, 8] in [0, 1]
            - direct_mode=False: 'sampled depths' [B, 8] (preferred) or
                                 'corner depths' [B, 8, H, W] / [B, 8]
        batch: dict containing:
            '3d_bbx': [B, 8, 3]
            'pad_left': [B,]
            'pad_top': [B,]
            'scale': [B,]
            'K': [B, 3, 3]
        Return:
            None, but updates internal IoU list
        '''
        pred_boxes = outputs['corner coords']
        if self.direct_mode:
            pred_depths = outputs['corner depths']
            if pred_depths.ndim != 2:
                raise ValueError(
                    f"Expected [B, 8] corner depths for direct_mode=True, got shape: {tuple(pred_depths.shape)}"
                )
            pred_depths = pred_depths * self.max_depth
        else:
            sampled_depths = outputs.get('sampled depths', None)
            if sampled_depths is not None and sampled_depths.ndim == 2:
                pred_depths = sampled_depths
            else:
                pred_depths = outputs['corner depths']
                if pred_depths.ndim == 4:
                    pred_depths = self.sample_depths(pred_depths, pred_boxes / 4.0)  # [B, 8]
                elif pred_depths.ndim != 2:
                    raise ValueError(f"Unexpected corner depths shape: {tuple(pred_depths.shape)}")
        gt_boxes = batch['3d_bbx']
        
        if isinstance(gt_boxes, list):
            gt_boxes = torch.stack([torch.as_tensor(x) for x in gt_boxes]).to(self.device).float()

        pred_boxes, pred_depths = self.cal_original(pred_boxes, pred_depths, batch["pad_left"], batch["pad_top"], batch["scale"], batch["K"], batch["h"])
        
        pred_boxes_3d_rectified = self._rectify_dispatch(
            pred_boxes, pred_depths, outputs.get('corner heatmaps', None), batch['K']
        )  # [B, 8, 3]
        
        # Compute IoU using PyTorch3D (Standard) and Invariant (Symmetry-Aware)
        ious_pytorch3d = self.iou3d_pytorch3d_safe(pred_boxes_3d_rectified, gt_boxes, verbose=False)
        ious_invariant = self.invariant_iou3d(pred_boxes_3d_rectified, gt_boxes)
        ious_hungarian = self.hungarian_iou3d(pred_boxes_3d_rectified, gt_boxes)
        
        # Take diagonal (pairwise match)
        self.iou_list.append(ious_pytorch3d.diagonal())
        self.iou_list_convex.append(ious_invariant.diagonal()) # Invariant returns matrix
        self.iou_list_hungarian.append(ious_hungarian.diagonal()) # Hungarian returns matrix
        
        B = pred_boxes.shape[0]
        self.total_num += B

    def _rectify_dispatch(self, pred_uv, pred_depths, heatmaps, K):
        if self.rectify_mode == "pca":
            return self.rectify_to_3d_box(pred_uv, pred_depths, heatmaps, K)
        if self.rectify_mode == "kabsch":
            corner_weights = None
            if self.use_corner_confidence:
                corner_weights = self._corner_weights_from_heatmaps(heatmaps)
            if self.infer_order_with_k:
                # Mirror the ordering inference used in moca_to_perldiff.py:
                # infer corner permutation from K+depth geometry (no GT usage).
                perm = self._infer_corner_permutation_from_k(pred_uv, pred_depths, K)
                pred_uv = self._reorder_by_perm(pred_uv, perm)
                pred_depths = self._reorder_by_perm(pred_depths, perm)
                if corner_weights is not None:
                    corner_weights = self._reorder_by_perm(corner_weights, perm)
            return self.rectify_to_3d_box_kabsch(pred_uv, pred_depths, K, corner_weights=corner_weights)
        raise ValueError(f"Unknown rectify_mode: {self.rectify_mode}. Use 'pca' or 'kabsch'.")

    def _unproject_corners(self, pred_uv, pred_depths, K, eps: float = 1e-6):
        fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
        u, v, z = pred_uv[..., 0], pred_uv[..., 1], pred_depths
        x = (u - cx[:, None]) * z / (fx[:, None] + eps)
        y = (v - cy[:, None]) * z / (fy[:, None] + eps)
        return torch.stack([x, y, z], dim=-1)  # [B, 8, 3]

    def _corner_weights_from_heatmaps(self, heatmaps, eps: float = 1e-3):
        if heatmaps is None:
            return None
        if heatmaps.ndim != 4:
            return None
        # Per-corner confidence from heatmap peak response; shift to keep strictly positive.
        peak = heatmaps.float().flatten(2).amax(dim=-1)  # [B, 8]
        peak = peak - peak.amin(dim=1, keepdim=True)
        return peak + eps

    def _infer_corner_permutation_from_k(self, pred_uv, pred_depths, K):
        # Same idea as utils.perldiff_wrapper.infer_ordered_uv_from_iou_logic:
        # 1) unproject original corners with K
        # 2) build IoU-rectified reference cuboid
        # 3) Hungarian-match original->reference to get ordered indices
        p_orig = self._unproject_corners(pred_uv, pred_depths, K)  # [B,8,3]
        p_ref = self.rectify_to_3d_box(pred_uv, pred_depths, None, K)  # [B,8,3]

        bsz = pred_uv.shape[0]
        perm = torch.zeros((bsz, 8), dtype=torch.long, device=pred_uv.device)
        for b in range(bsz):
            cost = torch.cdist(p_orig[b].unsqueeze(0), p_ref[b].unsqueeze(0)).squeeze(0)  # [8,8]
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            for r, c in zip(row_ind, col_ind):
                perm[b, c] = int(r)
        return perm

    def _reorder_by_perm(self, x, perm):
        if x.ndim == 3:
            idx = perm.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            return torch.gather(x, dim=1, index=idx)
        if x.ndim == 2:
            return torch.gather(x, dim=1, index=perm)
        raise ValueError(f"Unsupported tensor rank for corner reordering: {x.ndim}")

    def rectify_to_3d_box_kabsch(self, pred_uv, pred_depths, K, corner_weights=None, eps: float = 1e-6):
        """
        Fit a valid cuboid using weighted Procrustes/Kabsch while preserving corner correspondence.

        Args:
            pred_uv: [B, 8, 2]
            pred_depths: [B, 8]
            K: [B, 3, 3]
            corner_weights: optional [B, 8] corner confidences
        Returns:
            refined_P: [B, 8, 3]
        """
        B = pred_uv.shape[0]
        dtype = pred_uv.dtype
        device = pred_uv.device
        P = self._unproject_corners(pred_uv, pred_depths, K, eps=eps)  # [B, 8, 3]

        # Must match GT/label corner ordering.
        V = torch.tensor(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            dtype=dtype,
            device=device,
        ).unsqueeze(0).expand(B, -1, -1)  # [B, 8, 3]

        if corner_weights is None:
            w = torch.ones((B, 8), dtype=dtype, device=device)
        else:
            w = corner_weights.to(device=device, dtype=dtype).clamp_min(eps)
        w = w / (w.sum(dim=1, keepdim=True) + eps)  # [B, 8]
        w3 = w.unsqueeze(-1)  # [B, 8, 1]

        muP = (w3 * P).sum(dim=1, keepdim=True)  # [B, 1, 3]
        muV = (w3 * V).sum(dim=1, keepdim=True)  # [B, 1, 3]
        X = P - muP
        Y = V - muV

        # Row-vector formulation: solve Y @ R ~= X
        H = (w3 * Y).transpose(1, 2) @ X  # [B, 3, 3]
        U, _, Vh = torch.linalg.svd(H)
        R = U @ Vh

        # Enforce proper rotation det(R)=+1.
        det = torch.det(R)
        neg = det < 0
        if neg.any():
            U = U.clone()
            U[neg, :, -1] *= -1
            R = U @ Vh

        # Estimate anisotropic axis scales in local canonical frame.
        X_local = X @ R.transpose(1, 2)  # [B, 8, 3] ~= Y * s
        denom = (w3 * (Y ** 2)).sum(dim=1) + eps  # [B, 3]
        numer = (w3 * Y * X_local).sum(dim=1)  # [B, 3]
        s = (numer / denom).abs().clamp_min(eps)  # [B, 3]

        Y_scaled = Y * s.unsqueeze(1)  # [B, 8, 3]
        refined_P = (Y_scaled @ R) + muP  # [B, 8, 3]
        return refined_P
    
    def return_result(self):
        if self.total_num == 0:
            return 0.0
        all_ious = torch.cat(self.iou_list, dim=0)  # [N,]
        all_ious = torch.clamp(all_ious, max=1.0) 
        
        all_ious_invariant = torch.cat(self.iou_list_convex, dim=0)  # [N,]
        all_ious_invariant = torch.clamp(all_ious_invariant, max=1.0)
        
        all_ious_hungarian = torch.cat(self.iou_list_hungarian, dim=0)
        all_ious_hungarian = torch.clamp(all_ious_hungarian, max=1.0)
        
        avg_iou = all_ious.sum().item() / self.total_num
        max_iou = all_ious.max().item()
        
        avg_iou_invariant = all_ious_invariant.sum().item() / self.total_num
        max_iou_invariant = all_ious_invariant.max().item()
        
        avg_iou_hungarian = all_ious_hungarian.sum().item() / self.total_num
        max_iou_hungarian = all_ious_hungarian.max().item()
        
        # Using Invariant IoU for distribution stats as it's the intended robust metric
        total_ious = (all_ious_invariant <= 1.0).sum().item()
        print(f"IoU3D Results over {self.total_num} samples:")
        print(f"IoU = 0: {(all_ious_invariant == 0).sum().item() / total_ious}")
        print("-" * 50)
        print(f"[Standard]  Avg IoU3D: {avg_iou:.4f} | Max: {max_iou:.4f}")
        print(f"[Invariant] Avg IoU3D: {avg_iou_invariant:.4f} | Max: {max_iou_invariant:.4f}")
        print(f"[Hungarian] Avg IoU3D: {avg_iou_hungarian:.4f} | Max: {max_iou_hungarian:.4f}")
        return avg_iou, avg_iou_invariant
        
    def rectify_to_3d_box(self, pred_uv, pred_depths, heatmaps, K, inlier_scale=2.0):
        '''
        pred_uv: [B, 8, 2] 
        pred_depths: [B, 8]
        heatmaps: [B, 8, H, W]
        K: [B, 3, 3]
        Return:
            refined_P: [B, 8, 3]
        '''
        B = pred_uv.shape[0]
        eps = 1e-6
        
        fx, fy, cx, cy = K[:, 0,0], K[:, 1,1], K[:, 0,2], K[:, 1,2] # [B,]
        u, v, z = pred_uv[..., 0], pred_uv[..., 1], pred_depths # [B, 8]
        x = (u - cx[:, None]) * z / (fx[:, None] + eps) # [B, 8]
        y = (v - cy[:, None]) * z / (fy[:, None] + eps) # [B, 8]
        P = torch.stack([x, y, z], dim=-1) # [B, 8, 3]
        
        y_sorted_indices = torch.argsort(P[..., 1], dim=1, descending=True) # [B, 8]
        bottom_indices = y_sorted_indices[:, :4].unsqueeze(-1).expand(-1, -1, 3) # [B, 4, 3]
        top_indices = y_sorted_indices[:, 4:].unsqueeze(-1).expand(-1, -1, 3) # [B, 4, 3]
        
        P_bottom = torch.gather(P, dim=1, index=bottom_indices) # [B, 4, 3]
        P_top = torch.gather(P, dim=1, index=top_indices) # [B, 4, 3]
        
        up_vec = P_top.mean(dim=1) - P_bottom.mean(dim=1) # [B, 3]
        height = torch.norm(up_vec, dim=-1).clamp_min(eps) # [B,]
        up_axis = up_vec / height[:, None] # [B, 3]
        
        center = P.mean(dim=1) # [B, 3]
        
        # PCA
        P_centered = P - center[:, None, :] # [B, 8, 3]
        proj_dist = torch.einsum('bik, bk -> bi', P_centered, up_axis) # [B, 8]
        P_planar = P_centered - proj_dist.unsqueeze(-1) * up_axis.unsqueeze(1) # [B, 8, 3]
                
        ref_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(B, 3) # [B, 3]
        dot_check = torch.abs(torch.einsum('bk, bk -> b', ref_vec, up_axis)) # [B,]
        ref_vec_alt = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(B, 3) # [B, 3]
        ref_vec = torch.where(dot_check.unsqueeze(-1) > 0.99, ref_vec_alt, ref_vec) # [B, 3]
        
        basis_x = F.normalize(torch.cross(up_axis, ref_vec, dim=-1), dim=-1) # [B, 3]
        basis_z = F.normalize(torch.cross(basis_x, up_axis, dim=-1), dim=-1) # [B, 3]
        
        pts_2d_x = torch.einsum('bik, bk -> bi', P_planar, basis_x) # [B, 8]
        pts_2d_z = torch.einsum('bik, bk -> bi', P_planar, basis_z) # [B, 8]
        pts_2d = torch.stack([pts_2d_x, pts_2d_z], dim=-1) # [B, 8, 2]
        
        # SVD
        cov = torch.bmm(pts_2d.transpose(1,2), pts_2d) / 8.0
        U, S, V = torch.svd(cov)
        with torch.no_grad():
            detU = torch.linalg.det(U)
            flip = detU < 0
            if flip.any():
                U = U.clone()
                U[flip, :, 1] *= -1
    
        pts_on_axes = torch.bmm(pts_2d, U) # [B, 8, 2]
        max_pts = pts_on_axes.amax(dim=1) # [B, 2]
        min_pts = pts_on_axes.amin(dim=1) # [B, 2]
        extents = max_pts - min_pts # [B, 2]
        length, width = extents[:, 0], extents[:, 1]
        
        hl, hw, hh = (length / 2.0).unsqueeze(-1), (width / 2.0).unsqueeze(-1), (height / 2.0).unsqueeze(-1)
        local_temp = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1,1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype = torch.float32, device=self.device).unsqueeze(0) # [1, 8, 3]
        local_corners = local_temp * torch.stack([hl, hh, hw], dim=-1) # [B, 8, 3]
        basis_xz = torch.stack([basis_x, basis_z], dim=-1) # [B, 3, 2]
        R_yaw = torch.bmm(basis_xz, U.transpose(1,2)) # [B, 3, 2]
        R_final = torch.stack([R_yaw[..., 0], -up_axis, -R_yaw[..., 1]], dim=-1) # [B, 3, 3]
        
        with torch.no_grad():
            detR = torch.linalg.det(R_final)
            flip_r = detR < 0
            if flip_r.any():
                R_final = R_final.clone()
                R_final[flip_r, :, 2] *= -1
        
        refined_P = torch.bmm(local_corners, R_final.transpose(1,2)) + center.unsqueeze(1) # [B, 8, 3]
        return refined_P
    
    def invariant_iou3d(self, boxes_pred, boxes_gt):
        '''
        Compute 3D IoU checking valid rigid permutations to handle index ambiguity.
        Currently checks Identity and 180-degree rotation around Y-axis.
        This preserves box topology for iou_box3d, unlike random shuffling.
        
        boxes_pred: [B, 8, 3]
        boxes_gt: [B, 8, 3]
        '''
        B = boxes_pred.shape[0]
        
        # Permutation 1: Identity
        perm1 = [0, 1, 2, 3, 4, 5, 6, 7]
        
        # Permutation 2: 180-deg rotation around Y-axis (Top-Down axis)
        # Based on get_cuboid_verts_faces order: 
        # v0(-x,-y,-z) <-> v5(+x,-y,+z), v1 <-> v4, v2 <-> v7, v3 <-> v6
        perm2 = [5, 4, 7, 6, 1, 0, 3, 2]
        
        # Calculate IoU for Identity
        iou1 = self.iou3d_pytorch3d_safe(boxes_pred, boxes_gt, verbose=False)
        
        # Calculate IoU for Rotated
        boxes_pred_rot = boxes_pred[:, perm2, :]
        iou2 = self.iou3d_pytorch3d_safe(boxes_pred_rot, boxes_gt, verbose=False)
        
        # Take max
        ious = torch.max(iou1, iou2)
        return ious

    def hungarian_iou3d(self, boxes_pred, boxes_gt):
        '''
        Compute 3D IoU using Hungarian matching to reorder pred vertices to match GT topology.
        This handles "index swapping" and "bag of points" issues correctly.
        
        boxes_pred: [B, 8, 3]
        boxes_gt: [B, 8, 3]
        Return:
            ious: [B,] IoU for each pair
        '''
        B = boxes_pred.shape[0]
        # Store original pred for backup/comparison if needed, but we used reordered for IoU
        boxes_pred_reordered = boxes_pred.clone()
        
        for b in range(B):
            pred_b = boxes_pred[b]  # [8, 3]
            gt_b = boxes_gt[b]      # [8, 3]
            
            # Compute cost matrix (L2 distance)
            cost = torch.cdist(pred_b.unsqueeze(0), gt_b.unsqueeze(0)).squeeze(0)  # [8, 8]
            
            # Hungarian matching
            # row_ind: pred indices, col_ind: gt indices
            row_ind, col_ind = linear_sum_assignment(cost.cpu().detach().numpy())
            
            # Reorder pred to match GT order
            # If gt index j matches pred index i (pred[i] -> gt[j])
            # We want new_pred[j] = pred[i]
            for r, c in zip(row_ind, col_ind):
                boxes_pred_reordered[b, c] = pred_b[r]

        # Now compute standard box IoU with reordered vertices
        # We rely on PyTorch3D's robust implementation
        ious = self.iou3d_pytorch3d_safe(boxes_pred_reordered, boxes_gt, verbose=False)
        return ious

    def cal_original(self, pred_uv, pred_depths, pad_left, pad_top, scale, K, h):
        if not isinstance(pad_left, torch.Tensor):
            pad_left = torch.tensor(pad_left, device=pred_uv.device, dtype=torch.float32)
        if not isinstance(pad_top, torch.Tensor):
            pad_top = torch.tensor(pad_top, device=pred_uv.device, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=pred_uv.device, dtype=torch.float32)
            
        pred_uv = pred_uv.clone()
        pred_uv[..., 0] = (pred_uv[..., 0] - pad_left.view(-1, 1)) / scale.view(-1, 1) # [B, 8]
        pred_uv[..., 1] = (pred_uv[..., 1] - pad_top.view(-1, 1)) / scale.view(-1, 1) # [B, 8]
        virtual_scale = ((h / K[..., 1, 1]) * (self.f_v / self.h_v)).unsqueeze(1)
        pred_depths = pred_depths / virtual_scale
        return pred_uv, pred_depths

    def sample_depths(self, depth_maps, coords):
            """
            depth_maps: [B, 8, H, W]
            coords: [B, 8, 2] - (u,v) coordinates
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
        

    def _check_coplanar(self, boxes: torch.Tensor, eps: float = 1e-4) -> torch.BoolTensor:
        """
        boxes: (B, 8, 3)
        return: (B,) bool tensor
        """
        faces = torch.tensor(_box_planes, dtype=torch.int64, device=self.device)  # (6,4)
        verts = boxes.index_select(dim=1, index=faces.view(-1)).view(-1, faces.shape[0], faces.shape[1], 3)
        v0, v1, v2, v3 = verts.unbind(2)  # each: (B, 6, 3)

        e0 = F.normalize(v1 - v0, dim=-1)
        e1 = F.normalize(v2 - v0, dim=-1)
        normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)  # (B, 6, 3)

        # plane eq: (v3 - v0) · normal == 0
        dist = (v3 - v0) * normal
        dist = dist.sum(dim=-1).abs()  # (B, 6)

        return (dist < eps).all(dim=1)


    def _check_nonzero(self,boxes: torch.Tensor, eps: float = 1e-8) -> torch.BoolTensor:
        tris = torch.tensor(_box_triangles, dtype=torch.int64, device=self.device)  # (12,3)
        verts = boxes.index_select(dim=1, index=tris.view(-1)).view(-1, tris.shape[0], tris.shape[1], 3)
        v0, v1, v2 = verts.unbind(2)  # each: (B, 12, 3)

        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, 12, 3)
        face_areas = normals.norm(dim=-1) / 2.0          # (B, 12)

        return (face_areas > eps).all(dim=1)


    def iou3d_pytorch3d_safe(
        self,
        boxes_pred: torch.Tensor,
        boxes_gt: torch.Tensor,
        eps_coplanar: float = 1e-4,
        eps_nonzero: float = 1e-8,
        clamp_invalid_to_zero: bool = True,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Omni3D Style IoU3D computation with validity checks on predicted boxes.

        Args:
            boxes_pred: (N, 8, 3) predicted boxes corners
            boxes_gt:   (M, 8, 3) ground-truth boxes corners
        Returns:
            ious: (N, M)
        """
        assert boxes_pred.ndim == 3 and boxes_pred.shape[1:] == (8, 3)
        assert boxes_gt.ndim == 3 and boxes_gt.shape[1:] == (8, 3)

        boxes_pred = boxes_pred.float()
        boxes_gt = boxes_gt.float()
            
        # 1) validity checks on pred & GT (Omni3D checks dt)
        valid_coplanar_pred = self._check_coplanar(boxes_pred, eps=eps_coplanar)
        valid_nonzero_pred = self._check_nonzero(boxes_pred, eps=eps_nonzero)
        valid_pred = valid_coplanar_pred & valid_nonzero_pred
        
        valid_coplanar_gt = self._check_coplanar(boxes_gt, eps=eps_coplanar)
        valid_nonzero_gt = self._check_nonzero(boxes_gt, eps=eps_nonzero)
        valid_gt = valid_coplanar_gt & valid_nonzero_gt
        
        if verbose and (~valid_gt).any():
            n_bad_copl = int((~valid_coplanar_gt).sum().item())
            n_bad_nonz = int((~valid_nonzero_gt).sum().item())
            print(f"[IoU3D] Warning: invalid GT boxes -> IoU=0  | non-coplanar={n_bad_copl}, zero/degenerate={n_bad_nonz}")

        # 2) IoU compute (PyTorch3D C++/CUDA extension)
        # returns: (intersection_vol, iou) with shapes (N,M)
        _, ious = _C.iou_box3d(boxes_pred, boxes_gt)

        # 3) clamp invalid boxes to 0 IoU
        if clamp_invalid_to_zero and (~valid_pred).any():
            ious = ious.clone()
            ious[~valid_pred] = 0.0
            if verbose:
                n_bad_copl = int((~valid_coplanar_pred).sum().item())
                n_bad_nonz = int((~valid_nonzero_pred).sum().item())
                print(f"[IoU3D] Warning: invalid pred boxes -> IoU=0  | non-coplanar={n_bad_copl}, zero/degenerate={n_bad_nonz}")

        if clamp_invalid_to_zero and (~valid_gt).any():
            ious = ious.clone()
            ious[:, ~valid_gt] = 0.0
            
        return ious
        
