import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, PowerNorm
import math
import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import sys
import random
SRC_BETR_DIR = Path(__file__).resolve().parents[1]
PROJ_ROOT = SRC_BETR_DIR.parent
sys.path.insert(0, str(SRC_BETR_DIR))

from data.data_utils import _LetterBoxing

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def visualization(small_batch, outputs, out_dir: Path):
    pred_corners_all = outputs['corner coords'].detach().cpu().numpy()  # [B, 8, 2]
    gt_corners_all = (small_batch['gt_corners'].cpu().numpy()) * 512.0 # [B, 8, 2]
    # Unnormalize predictions
    dataset_root = PROJ_ROOT / "datasets"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(pred_corners_all.shape[0]):
        # Convert tensor to image
        image_path = dataset_root / small_batch["path"][i]
        image = Image.open(image_path).convert("RGB")
        image = _LetterBoxing(image, 512)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # load gt, predicted corners
        gt_corners = gt_corners_all[i]
        pred_corners = pred_corners_all[i]
        # Draw GT, Predicted corners
        for j in range(8):
            gt_u, gt_v = int(round(gt_corners[j][0])), int(round(gt_corners[j][1]))
            pd_u, pd_v = int(round(pred_corners[j][0])), int(round(pred_corners[j][1]))
            cv2.circle(image, (gt_u, gt_v), 5, (0, 255, 0), -1) # Green for GT
            cv2.circle(image, (pd_u, pd_v), 5, (0, 0, 255), -1) # Red for Pred
        cv2.imwrite(str(out_dir / f"vis_{i:03d}.png"), image)
    print(f"Visualization images saved to: {out_dir}")


def set_seed(seed: int, rank: int = 0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic settings
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def reduce_dict(input_dict, world_size, average=True):
    if not input_dict:
        return {}
    with torch.no_grad():
        names = sorted(input_dict.keys())
        values = [input_dict[k] for k in names]
        metrics_tensor = torch.tensor(values).cuda() # Move to GPU
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        if average:
            metrics_tensor /= world_size
        return {k: v.item() for k, v in zip(names, metrics_tensor)}


def get_scheduler(optimizer, cfg, num_batches_per_epoch):
    warmup_epochs = cfg["warmup_epochs"]
    num_epochs = cfg["num_epochs"]
    def lr_lambda(current_step):
        current_epoch = current_step / num_batches_per_epoch
        # Linear warmup
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        # Cosine annealing
        progress = float(current_epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def print_epoch_stats(epoch, num_epochs, train_metrics, val_metrics=None, ema_metrics=None):
    print("\n" + "="*85)
    print(f" 📊 Epoch [{epoch+1:03d}/{num_epochs:03d}] Summary")
    print("-" * 85)
    print(f" {'Mode':<10} | {'Total':<8} | {'Corners':<8} | {'Depths':<8}")
    print("-" * 85)
    
    # Train
    t = train_metrics
    print(f" {'Train':<10} | {t['total_loss']:.6f} | {t['loss_corners']:.6f} | {t['loss_depths']:.6f}")
    
    # Val
    if val_metrics:
        v = val_metrics
        print(f" {'Validation':<10} | {v['total_loss']:.6f} | {v['loss_corners']:.6f} | {v['loss_depths']:.6f}")
    if ema_metrics:
        e = ema_metrics
        print(f" {'EMA Val':<10} | {e['total_loss']:.6f} | {e['loss_corners']:.6f} | {e['loss_depths']:.6f}")
    print("="*85 + "\n")


def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _select_representative_corners(gt_coords, total_corners, num_corners):
    available = min(total_corners, gt_coords.shape[0])
    if available == 0:
        return []

    num_corners = min(max(int(num_corners), 1), available)
    if num_corners == available:
        return list(range(available))
    if num_corners == 1:
        return [0]

    coords = gt_coords[:available]
    pairwise = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(pairwise, axis=-1)
    np.fill_diagonal(dists, -np.inf)

    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    selected = [int(i), int(j)]

    while len(selected) < num_corners:
        min_dist_to_selected = np.min(dists[:, selected], axis=1)
        min_dist_to_selected[selected] = -np.inf
        selected.append(int(np.argmax(min_dist_to_selected)))

    return selected


def visualize_heatmaps(
    heatmaps,
    pred_coords_128,
    gt_coords_128,
    save_path,
    selected_corners=None,
    num_corners=4,
    cmap="jet",
    value_range=(0.0, 1.0),
    grid_shape=(2, 2),
    square_grid=True,
    show_titles=False,
    save_aux_assets=False,
    norm_gamma=1.25,
    auto_percentiles=(0.05, 99.95),
):
    """
    Paper-oriented heatmap visualization.
    heatmaps: [N, H, W]
    pred_coords_128: [N, 2]
    gt_coords_128: [N, 2]
    selected_corners: optional explicit corner indices
    num_corners: number of representative corners to display when selected_corners is None
    grid_shape: default subplot grid size as (rows, cols)
    show_titles: if True, show subplot title text
    save_aux_assets: if True, save legend/colorbar as separate image files
    norm_gamma: gamma for heatmap tone mapping (1.0 means linear)
    auto_percentiles: (low, high) percentile range for auto value_range when value_range=None
    """
    heatmaps_np = _to_numpy(heatmaps)
    pred_coords_np = _to_numpy(pred_coords_128)
    gt_coords_np = _to_numpy(gt_coords_128)

    if heatmaps_np.ndim != 3:
        raise ValueError(f"heatmaps must be [N, H, W], got {heatmaps_np.shape}")

    total_corners = min(heatmaps_np.shape[0], pred_coords_np.shape[0], gt_coords_np.shape[0])
    if total_corners == 0:
        raise ValueError("No valid corners found for visualization.")

    if selected_corners is None:
        corner_indices = _select_representative_corners(gt_coords_np, total_corners, num_corners)
    else:
        corner_indices = []
        used = set()
        for idx in selected_corners:
            idx = int(idx)
            if 0 <= idx < total_corners and idx not in used:
                corner_indices.append(idx)
                used.add(idx)
        if not corner_indices:
            raise ValueError("selected_corners does not contain a valid corner index.")

    if value_range is not None:
        vmin, vmax = float(value_range[0]), float(value_range[1])
    else:
        shown_heatmaps = heatmaps_np[corner_indices]
        # Robust auto-stretch for better contrast while ignoring extreme outliers.
        p_low = float(auto_percentiles[0])
        p_high = float(auto_percentiles[1])
        p_low = min(max(p_low, 0.0), 100.0)
        p_high = min(max(p_high, p_low + 1e-6), 100.0)
        vmin = float(np.percentile(shown_heatmaps, p_low))
        vmax = float(np.percentile(shown_heatmaps, p_high))
        if np.isclose(vmin, vmax):
            vmin = float(np.min(shown_heatmaps))
            vmax = float(np.max(shown_heatmaps))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    if norm_gamma is not None and not np.isclose(float(norm_gamma), 1.0):
        heatmap_norm = PowerNorm(gamma=float(norm_gamma), vmin=vmin, vmax=vmax)
    else:
        heatmap_norm = Normalize(vmin=vmin, vmax=vmax)

    n_plots = len(corner_indices)
    if grid_shape is not None:
        n_rows = max(int(grid_shape[0]), 1)
        n_cols = max(int(grid_shape[1]), 1)
        capacity = n_rows * n_cols
        if n_plots > capacity:
            n_rows = int(np.ceil(n_plots / n_cols))
    elif square_grid:
        n_side = int(np.ceil(np.sqrt(n_plots)))
        n_rows, n_cols = n_side, n_side
    else:
        n_rows, n_cols = 1, n_plots

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.15 * n_cols, 3.15 * n_rows),
        dpi=220,
    )
    axes = np.atleast_1d(axes).reshape(-1)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    im = None
    for axis, corner_idx in zip(axes, corner_indices):
        h = heatmaps_np[corner_idx]
        im = axis.imshow(h, cmap=cmap, origin="upper", norm=heatmap_norm)

        gt_x, gt_y = gt_coords_np[corner_idx]
        axis.scatter(
            gt_x,
            gt_y,
            marker="o",
            s=90,
            facecolors="none",
            edgecolors="#00a65a",
            linewidths=2.0,
            zorder=3,
        )

        pred_x, pred_y = pred_coords_np[corner_idx]
        axis.scatter(
            pred_x,
            pred_y,
            marker="x",
            s=90,
            c="#d62d20",
            linewidths=2.2,
            zorder=3,
        )

        if show_titles:
            axis.set_title(f"Corner {corner_idx}", fontsize=11, pad=3)
        axis.set_aspect("equal")
        axis.axis("off")

    for axis in axes[n_plots:]:
        axis.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.8,
            markeredgecolor="#00a65a",
            linestyle="None",
            label="GT",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            markersize=7,
            markeredgewidth=2.0,
            color="#d62d20",
            linestyle="None",
            label="Pred",
        ),
    ]

    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.006, top=0.994, wspace=0.006, hspace=0.006)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    if save_aux_assets:
        legend_path = save_path.with_name(f"{save_path.stem}_legend{save_path.suffix}")
        colorbar_path = save_path.with_name(f"{save_path.stem}_colorbar{save_path.suffix}")

        legend_fig, legend_ax = plt.subplots(figsize=(2.2, 0.6), dpi=300)
        legend_ax.axis("off")
        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=10,
            columnspacing=0.9,
            handletextpad=0.35,
        )
        legend_fig.savefig(legend_path, bbox_inches="tight", pad_inches=0.01, transparent=True)
        plt.close(legend_fig)

        if im is not None:
            colorbar_fig, colorbar_ax = plt.subplots(figsize=(0.5, 2.8), dpi=300)
            colorbar_fig.subplots_adjust(left=0.34, right=0.76, bottom=0.06, top=0.98)
            sm = plt.cm.ScalarMappable(norm=heatmap_norm, cmap=cmap)
            sm.set_array([])
            cbar = colorbar_fig.colorbar(sm, cax=colorbar_ax)
            cbar.ax.tick_params(labelsize=8, length=2)
            colorbar_fig.savefig(colorbar_path, bbox_inches="tight", pad_inches=0.01, transparent=True)
            plt.close(colorbar_fig)
  
    
def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), lr_multiplier=None):
    if lr_multiplier is None:
        lr_multiplier = {}
    parameter_group_vars = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1. Determine Weight Decay Group
        if len(param.shape) == 1 or name.endswith(".bias") or "position_encoding" in name or "box_embedding" in name or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        # 2. Determine LR Multiplier
        this_lr_scale = 1.0
        for search_key, scale in lr_multiplier.items():
            if search_key in name:
                this_lr_scale = scale
                break
        # Create a unique group name based on both decay and lr_scale
        full_group_name = f"{group_name}_lr{this_lr_scale}"
        if full_group_name not in parameter_group_vars:
            parameter_group_vars[full_group_name] = {
                "params": [], 
                "weight_decay": this_weight_decay,
                "lr_scale": this_lr_scale
            }
        parameter_group_vars[full_group_name]["params"].append(param)
    return list(parameter_group_vars.values())


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    # mp.set_sharing_strategy('file_system')
    torch.cuda.set_device(rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=rank)
    except:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cudnn.rnn.fp32_precision = "tf32"
    torch.set_float32_matmul_precision('high')


def print_grad_analysis(model, criterion, loss_dict):
    is_ddp = hasattr(model, 'module')
    actual_model = model.module if is_ddp else model
    
    target_modules = {
        "Encoder": actual_model.transformer_encoder,
        "Transformer Decoder": actual_model.transformer_decoder,
        # "DA Embedder": actual_model.da_embedder,
        "Upsample Layer": actual_model.upsample,
        "Box Embedding": actual_model.box_embedding,
    }
    if hasattr(actual_model, "prediction_heads"):
        target_modules["Prediction Heads"] = actual_model.prediction_heads
    elif hasattr(actual_model, "direct_head"):
        target_modules["Direct Head"] = actual_model.direct_head
    else:
        if hasattr(actual_model, "heatmap_head"):
            target_modules["Heatmap Head"] = actual_model.heatmap_head
        if hasattr(actual_model, "corner_depth_head"):
            target_modules["Depth Head"] = actual_model.corner_depth_head
    # Define loss components to analyze (applying weights)
    loss_components = {
        "Corner": loss_dict["loss_corners"],
        "Depth": loss_dict["loss_depths"] * criterion.weight_depth
    }
    print("\n" + "="*100)
    header = f"{'Module Name':<25} | " + " | ".join([f"{k:<12}" for k in loss_components.keys()]) + " | Total"
    print(header)
    print("="*100)
    grad_data = {m_name: {l_name: 0.0 for l_name in loss_components} for m_name in target_modules}
    
    # Calculate individual component gradients WITHOUT DDP sync
    # These are per-rank gradients that need manual averaging later
    for l_name, l_val in loss_components.items():
        model.zero_grad()
        if l_val.requires_grad:
            l_val.backward(retain_graph=True)
            # Calculate gradients per rank
            for m_name, module in target_modules.items():
                norm = 0.0
                for p in module.parameters():
                    if p.grad is not None:
                        norm += p.grad.data.norm(2).item() ** 2
                grad_norm = norm ** 0.5
                grad_data[m_name][l_name] = grad_norm
    
    # Final backward on total_loss to populate actual gradients for optimizer
    model.zero_grad()
    loss_dict["total_loss"].backward(retain_graph=True)   
    
    # Calculate total gradient norms (these are already averaged by DDP)
    total_norms = {}
    for m_name, module in target_modules.items():
        total_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norms[m_name] = total_norm ** 0.5
    
    # Print results
    for m_name in target_modules:
        row = f"{m_name:<25} | "
        row += " | ".join([f"{grad_data[m_name][l_name]:<12.4f}" for l_name in loss_components.keys()])
        row += f" | {total_norms[m_name]:.4f}"
        print(row)

    total_grad_norm = 0.0
    for p in actual_model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    print("="*100)
    print(f"Total Model Grad Norm: {total_grad_norm**0.5:.4f}")
    print("="*100 + "\n")
    
    model.zero_grad(set_to_none=True)
