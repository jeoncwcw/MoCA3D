from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file as load_safetensors

ROOT_BETR = Path(__file__).parent.parent.parent
sys.path.insert(0, f"{ROOT_BETR}/models")


from dinov3.vision_transformer import vit_large

# --------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------

def _pick_device(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------------
# DINOv3 Feature Extractor
# --------------------------------------------------------------------------
DINOV3_CKPT= ROOT_BETR / "checkpoints" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

def _load_dinov3_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _infer_ffn_layer(state_dict: dict) -> str:
    if any(".ffn.w1.weight" in k for k in state_dict):
        return "swiglu"
    return "mlp"


def _infer_storage_tokens(state_dict: dict) -> int:
    for key, value in state_dict.items():
        if "storage_tokens" in key and isinstance(value, torch.Tensor):
            if value.ndim == 3:
                return int(value.shape[1])
            return int(value.numel())
    return 0


def _infer_layerscale_init(state_dict: dict) -> float | None:
    has_layerscale = any((".ls1.gamma" in k) or (".ls2.gamma" in k) for k in state_dict)
    return 1e-5 if has_layerscale else None


class DINOv3FeatureExtractor:
    """Wrapper for extracting final block tokens from a local DINOv3 ViT-L/16 checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path = DINOV3_CKPT,
        device: str | torch.device | None = "auto",
        patch_size: int = 16,
        ffn_layer: str | None = None,
        storage_tokens: int | None = None,
        layerscale_init: float | None = None,
    ) -> None:
        self.device = _pick_device(device)
        state = _load_dinov3_checkpoint(checkpoint_path)

        ffn = ffn_layer or _infer_ffn_layer(state)
        storage = storage_tokens if storage_tokens is not None else _infer_storage_tokens(state)
        ls_init = layerscale_init if layerscale_init is not None else _infer_layerscale_init(state)
        if ls_init == 0:
            ls_init = None

        self.model = vit_large(
            patch_size=patch_size,
            ffn_layer=ffn,
            layerscale_init=ls_init,
            n_storage_tokens=storage,
        ).to(self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # if missing:
        #     print(f"DINOv3 missing keys {len(missing)} (first 5): {list(missing)[:5]}")
        # if unexpected:
        #     print(f"DINOv3 unexpected keys {len(unexpected)} (first 5): {list(unexpected)[:5]}")
        self.model.eval()

    def __call__(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: processed image batch shaped (B, C, H, W) or single image (C, H, W).
        Returns:
            Patch tensor shaped (B, C, H_tokens, W_tokens) on device.
        """
        patch_h = self.model.patch_embed.patch_size
        if isinstance(patch_h, tuple):
            patch_h, patch_w = patch_h
        else:
            patch_w = patch_h
        if images.dim() == 3:
            batch = images.unsqueeze(0)
        elif images.dim() == 4:
            batch = images
        else:
            raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(images.shape)}")
        batch = batch.to(self.device)

        with torch.no_grad():
            feat_dict = self.model.forward_features(batch)
            patch_tokens = feat_dict["x_norm_patchtokens"]  # (B, N, C)


        h_tokens = batch.shape[-2] // patch_h
        w_tokens = batch.shape[-1] // patch_w
        expected = h_tokens * w_tokens
        if expected != patch_tokens.shape[1]:
            h_tokens = int(math.sqrt(patch_tokens.shape[1]))
            w_tokens = patch_tokens.shape[1] // h_tokens

        patch_grid = patch_tokens.reshape(
            patch_tokens.shape[0], h_tokens, w_tokens, patch_tokens.shape[-1]
        ).permute(0, 3, 1, 2)

        return patch_grid

# Convenience helpers: one-line predict wrappers using local checkpoints
# --------------------------------------------------------------------------

def dinov3_predict(
    images: torch.Tensor,
    checkpoint_path: Path = DINOV3_CKPT,
    device: str | torch.device | None = "auto",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Extract final block tokens from a local DINOv3 checkpoint.
    """
    extractor = DINOv3FeatureExtractor(checkpoint_path=checkpoint_path, device=device)
    return extractor(images)
