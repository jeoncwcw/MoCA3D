import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from models.feature_modules import DINOv3FeatureExtractor
from models.transformer_modules import (
    TransformerEncoder, TransformerDecoder, 
    build_position_encoding, BoxEmbedding, _prepare_mask_for_transformer, _unpatchify,
)
from models.dense_modules import DenseHeads, UpsampleLayer, SoftArgmax2D

from models.transformer_modules.encoder import TransformerEncoderLayer
from models.transformer_modules.decoder import TransformerDecoderLayer
import torch
from torch import nn
import torch.nn.functional as F

def _ensure_non_empty_keep_mask(keep_mask):
    B, _, H, W = keep_mask.shape
    flat = keep_mask.flatten(1)
    empty = flat.sum(dim=1) == 0
    # Compile-friendly fallback: for empty samples only, set one random token to True.
    rnd = torch.randint(0, H * W, (B, 1), device=keep_mask.device)
    fallback = torch.zeros_like(flat)
    fallback.scatter_(1, rnd, True)
    flat = torch.where(empty.unsqueeze(1), fallback, flat)
    return flat.view(B, 1, H, W)

class Moca3DModel(nn.Module):
    def __init__(self, cfg):
        super(Moca3DModel, self).__init__()
        # Feature Extractors
        self.feature_mode = cfg.feature_mode
        if not self.feature_mode:
            print("[Image Mode] Loading Backbones ...")
            self.feature_dinov3 = DINOv3FeatureExtractor(
                checkpoint_path=Path(cfg.dinov3_checkpoint_path),
                device=cfg.device,
            )
        else:
            print("[Feature Mode] Skipping Backbone Loading ...")
            self.feature_monodepth = None
            self.feature_metricdepth = None
            self.feature_dinov3 = None
            
        # Feature Generator
        self.dino_channels = 1024
        self.conv1x1 = nn.Conv2d(in_channels=self.dino_channels, out_channels=cfg.d_model, kernel_size=1)

        # Transformer Encoder and Decoder
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.encoder.nhead,
            dim_feedforward=cfg.encoder.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, cfg.encoder.num_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.decoder.nhead,
            dim_feedforward=cfg.decoder.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, cfg.decoder.num_layers)

        # Positional Encoding and Box Embedding
        self.position_encoding = build_position_encoding(
            hidden_dim=cfg.d_model,
            temperature= cfg.position_encoding.temperature,
            normalize=cfg.position_encoding.normalize,
            scale=cfg.position_encoding.scale,
        )
        self.box_embedding = BoxEmbedding(
            d_model=cfg.d_model,
            temperature=cfg.box_embedding.temperature,
            scale=cfg.box_embedding.scale,
        )

        # Upsample and Prediction Heads
        # skip_feature is produced from encoder tokens with channel=d_model
        self.upsample = UpsampleLayer(d_model=cfg.d_model, skip_channels=cfg.d_model, activation=cfg.activation)
        self.prediction_heads = DenseHeads(
            heads=cfg.prediction_heads,
            in_channels=cfg.d_model // 4,  # After two upsampling layers
        )    
        self.soft_argmax = SoftArgmax2D(beta=cfg.soft_argmax.beta, is_sigmoid=cfg.soft_argmax.is_sigmoid)
        self.heatmap_size = int(cfg.heatmap_size)
        self.input_size = int(cfg.input_size)
        self.hm_to_img_scale = self.input_size / float(self.heatmap_size)
        
        # Masking setup
        self.setup_token_masking(cfg)
        self.setup_prior_bbox(cfg) # Setup box prior parameters and layers
    
    def setup_token_masking(self, cfg):
        self.token_mask_prob = cfg.aug_token.token_mask_prob
        self.mask_block_prob = cfg.aug_token.get("mask_block_prob", 0.0)
        self.modality_dropout_prob = cfg.aug_token.get("modality_dropout_prob", 0.0)
        self.ex_masked_token_attn = cfg.aug_token.get("exclude_masked_token_attn", True)
        self.block_mask_hw = (cfg.aug_token.get("block_mask_hw", 2), cfg.aug_token.get("block_mask_hw", 2))
        self.mask_scale_by_keep_ratio = cfg.aug_token.get("mask_scale_by_keep_ratio", False) 
        
    def setup_prior_bbox(self, cfg):
        self.use_box_prior = cfg.aug_box.get("use_box_prior", False)
        self.box_prior_clamp = float(cfg.aug_box.get("box_prior_clamp", 4.0))
        self.box_prior_dropout = float(cfg.aug_box.get("box_prior_dropout", 0.0))
        prior_in_ch = int(cfg.aug_box.get("box_prior_ch", 4))
        
        self.box_prior_proj = nn.Sequential(
            nn.Conv2d(prior_in_ch, cfg.d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=1)
        )
        
        init_logit = float(cfg.aug_box.get("box_prior_init_logit", -3.0))
        self.box_prior_alpha_logit = nn.Parameter(torch.tensor(init_logit)) # Learnable Soft Gate
        if not self.use_box_prior:
            self.box_prior_proj.requires_grad_(False)
            self.box_prior_alpha_logit.requires_grad_(False)
        
    def forward(self, bbx2d_tight, mask = None,
                images_dino = None, f_dino = None,
                return_decoder_feat: bool = False,
                ):
        # Generate combined features
        try:
            if self.feature_mode:
                dinov3_features = f_dino
            else:
                self.feature_dinov3.model.eval()
                with torch.no_grad():
                    dinov3_features = self.feature_dinov3(images_dino)
        except RuntimeError as e:
            print("RuntimeError in feature extraction:", e)
            raise e
        dinov3_features = self.conv1x1(dinov3_features) # [B, C, H, W]
        
        # Provide relative box positional prior
        if self.use_box_prior:
            B, _, H, W = dinov3_features.shape
            box_prior = self._build_box_prior_map(bbx2d_tight, H, W, dinov3_features.device, dinov3_features.dtype)
            if self.training and self.box_prior_dropout > 0.0:
                drop_prior = torch.rand((), device=dinov3_features.device) < self.box_prior_dropout
                box_prior = torch.where(drop_prior, torch.zeros_like(box_prior), box_prior)
            prior_emb = self.box_prior_proj(box_prior)
            alpha = torch.sigmoid(self.box_prior_alpha_logit)
            dinov3_features = dinov3_features + alpha * prior_emb
        skip_feature = dinov3_features.clone()
        
        if self.training:
            B, _, H, W = dinov3_features.shape
            dino_token_keep_mask =  None
            if self.token_mask_prob > 0.0:
                use_block_gate = torch.rand((), device=dinov3_features.device) < self.mask_block_prob
                dino_token_keep_mask = self._sample_spatial_keep_mask(
                    B, H, W, dinov3_features.device, self.token_mask_prob, use_block_gate
                )
                dinov3_features = self._apply_feature_mask(dinov3_features, dino_token_keep_mask)
        
        # Add positional encoding
        dinov3_pos_encoding = self.position_encoding(dinov3_features)
        padding_mask = _prepare_mask_for_transformer(mask)
        box_embeddings = self.box_embedding(bbx2d_tight)
        
        padding_mask_dino = padding_mask
        if self.training and self.ex_masked_token_attn:
            if dino_token_keep_mask is not None:
                token_drop_dino = ~dino_token_keep_mask.squeeze(1).flatten(1)
                padding_mask_dino = padding_mask_dino | token_drop_dino

        # Permuting for transformer input
        dinov3_features = dinov3_features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)    
        dinov3_pos_encoding = dinov3_pos_encoding.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        box_embeddings = box_embeddings.permute(1, 0, 2)  # (9, B, C)
        
        # Pass through Transformer and unpatchify
        image_feat = self.transformer_encoder(dinov3_features, image_feat_key_padding_mask=padding_mask_dino, pos=dinov3_pos_encoding)
        decoder_feat = self.transformer_decoder(
            image_feat,
            box_embeddings,
            image_feat_key_padding_mask=padding_mask_dino,
            image_feat_pos=dinov3_pos_encoding,
        )
        decoder_feat = _unpatchify(decoder_feat) # (B, C, H, W)

        # Upsample, Prediction Heads, Soft Argmax and get center coords
        output = self.upsample(decoder_feat, skip_feature)
        output = self.prediction_heads(output)

        valid_mask = 1.0 - F.interpolate(mask.unsqueeze(1).float(), size=(self.heatmap_size, self.heatmap_size), mode="nearest") # [B, 1, 128, 128]
        valid_mask = valid_mask.to(output['corner heatmaps'].device)

        output['corner heatmaps'] = output['corner heatmaps'] * valid_mask # [B, 8, 128, 128]
        corner_coords = self.soft_argmax(output['corner heatmaps']) # [B, 8, 2]
        output['corner coords'] = corner_coords * self.hm_to_img_scale # Scale back to original image coordinates
        output['sampled depths'] = self._sample_depths(output['corner depths'], corner_coords)
        if return_decoder_feat:
            output['decoder_feat'] = decoder_feat
            output['2d_bbx'] = bbx2d_tight
        
        return output
    
    def _sample_block_keep_mask(self, B, H, W, device, mask_prob):
        if mask_prob <= 0.0:
            return torch.ones((B, 1, H, W), dtype=torch.bool, device=device)
        
        bh = min(self.block_mask_hw[0], H)
        bw = min(self.block_mask_hw[1], W)
        valid_h = max(H - bh + 1, 1)
        valid_w = max(W - bw + 1, 1)
        
        gamma = (mask_prob * H  * W) / float(bh * bw * valid_h * valid_w)
        gamma = max(0.0, min(gamma, 1.0))
        
        seeds = (torch.rand((B, 1, H, W), device=device) < gamma).float()
        dropped = F.max_pool2d(
            seeds, kernel_size=(bh, bw), stride=1, padding=(bh//2, bw//2)
        )
        dropped = dropped[..., :H, :W] > 0
        keep_mask = ~dropped
        return keep_mask
    
    def _sample_spatial_keep_mask(self, B, H, W, device, mask_prob, use_block_gate):
        keep_block = self._sample_block_keep_mask(B, H, W, device, mask_prob)
        keep_rand = (torch.rand((B, 1, H, W), device=device) >= mask_prob)
        gate = torch.as_tensor(use_block_gate, device=device, dtype=torch.bool).view(1, 1, 1, 1)
        keep = torch.where(gate, keep_block, keep_rand)
        return _ensure_non_empty_keep_mask(keep)
    
    def _apply_feature_mask(self, feat, keep_mask):
        keep = keep_mask.to(feat.dtype)
        if self.mask_scale_by_keep_ratio:
            keep_ratio = keep.mean(dim=[1,2,3], keepdim=True).clamp(min=1e-6)
            return feat * keep / keep_ratio
        return feat * keep
    
    def _build_box_prior_map(self, bbx2d_norm, H, W, device, dtype):
        x1, y1, x2, y2 = bbx2d_norm.unbind(dim=-1)
        eps = 1e-6
        w = (x2 - x1).clamp(min=eps)
        h = (y2 - y1).clamp(min=eps)
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / float(H)
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / float(W)
        try:
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        except TypeError:
            yy, xx = torch.meshgrid(ys, xs)
        
        B = bbx2d_norm.shape[0]
        xx = xx[None, None, :, :].expand(B, 1, H, W)
        yy = yy[None, None, :, :].expand(B, 1, H, W)
        
        x1v, y1v = x1[:, None, None, None], y1[:, None, None, None]
        wv, hv = w[:, None, None, None], h[:, None, None, None]
        xcv, ycv = xc[:, None, None, None], yc[:, None, None, None]
        
        dx, dy = (xx - xcv) / wv, (yy - ycv) / hv
        u, v = (xx - x1v) / wv, (yy - y1v) / hv
        
        prior = torch.cat([dx, dy, u, v], dim=1)
        if self.box_prior_clamp > 0.0:
            prior = prior.clamp(min=-self.box_prior_clamp, max=self.box_prior_clamp)
        return prior
    
    def _sample_depths(self, depth_maps, coords):
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
        flat_grid = norm_coords.view(B * num_corners, 1, 1, 2)
        
        sample_depths = F.grid_sample(flat_depth_maps, flat_grid, mode="bilinear", align_corners=True)
        return sample_depths.view(B, num_corners)
