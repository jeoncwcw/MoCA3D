import torch
import copy
from torch.nn import functional as F

def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

def _prepare_mask_for_transformer(masks, patch_size=16):
    masks_float = masks.unsqueeze(1).float()  # (B, 1, H, W)
    patch_padding_ratio = F.avg_pool2d(
        masks_float,
        kernel_size=patch_size,
        stride=patch_size,
        ceil_mode=True,
    )
    patch_mask = (patch_padding_ratio == 1.0)
    return patch_mask.flatten(1).bool()  # (B, N_patches)

def _unpatchify(x):
    x = x.permute(1, 0, 2) 
    B, N, C = x.shape
    H = W = int((N) ** 0.5)

    x_2d = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
    return x_2d