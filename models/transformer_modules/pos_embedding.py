import torch
from torch import nn
import math

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temprerature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temprerature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
class BoxEmbedding(nn.Module):
    def __init__(self, d_model=256, temperature=10000, scale=None):
        super(BoxEmbedding, self).__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.scale = scale if scale is not None else 2 * math.pi
        self.num_pos_feats = d_model // 2
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.id_embed = nn.Embedding(9, d_model)

    def forward(self, box_corners):
        """
        Args:
            box_corners: [Batch, 4] tensor (Normalized x1, y1, x2, y2)
        Returns:
            output: [Batch, 4, hidden_dim]
        """
        B, _ = box_corners.shape
        x1, y1, x2, y2 = box_corners.unbind(-1)
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2

        all_comps = torch.stack([x1, y1, x2, y2, xm, ym], dim=-1) 
        idx = [0, 1, 2, 1, 0, 3, 2, 3, 4, 1, 4, 3, 0, 5, 2, 5, 4, 5]
        key_points = all_comps[:, idx].view(B, 9, 2)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=box_corners.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        coords_scaled = key_points * self.scale # [B, 9, 2]
        pos = coords_scaled.unsqueeze(-1) / dim_t  # [B, 9, 2, num_pos_feats]
        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=4).flatten(3)  # [B, 9, num_pos_feats]
        pos = pos.flatten(2) # [B, 9, d_model]
        output = self.mlp(pos) + self.id_embed.weight.unsqueeze(0)
        return output


def build_position_encoding(hidden_dim=256, temperature=10000, normalize=False, scale=None):
    return PositionEmbeddingSine(
        num_pos_feats=hidden_dim // 2,
        temprerature=temperature,
        normalize=normalize,
        scale=scale,
    )