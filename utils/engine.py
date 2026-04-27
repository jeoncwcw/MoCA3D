from pathlib import Path
import torch
from typing import Iterable
from collections import defaultdict
from tqdm import tqdm
from .functions import reduce_dict, visualize_heatmaps, print_grad_analysis
from typing import Any

def train_one_epoch(
    model: torch.nn.Module,
    model_ema: torch.nn.Module,
    train_dataloader: Iterable,
    rank: int, epoch: int, num_batches_per_epoch: int,
    device: torch.device,
    criterion: torch.nn.Module,
    grad_accum_steps: int,
    scaler: torch.cuda.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    world_size: int,
):
    model.train()
    criterion.set_epoch(epoch)
    train_meter = defaultdict(float)
    train_num_samples = 0
    iterator = train_dataloader
    if rank == 0:
        iterator = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}] Training", leave=True, total=num_batches_per_epoch)
    for i, batch in enumerate(iterator):
        # if i % 500 == 0:
        #     import gc
        #     gc.collect()
        #     torch.cuda.empty_cache()
        batch_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.amp.autocast('cuda'):
            outputs = model(
                bbx2d_tight=batch_gpu["2d_bbx"],
                mask=batch_gpu["padding_mask"],
                # f_metric=batch_gpu["feat_metric"],
                # f_mono=batch_gpu["feat_mono"],
                # f_dino=batch_gpu["feat_dino"],
                images_dino = batch_gpu['image_dino']
            )
            loss_dict = criterion(outputs, batch_gpu)
            total_loss = loss_dict["total_loss"] / grad_accum_steps
        # Disabled: gradient analysis causes accumulation artifacts with composite losses
        # if (i ) % 300 == 0 and rank == 0:
        #     print_grad_analysis(model, criterion, loss_dict)
        scaler.scale(total_loss).backward()
        if (i + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            model_ema.update(model)
            scheduler.step()
        with torch.no_grad():
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    train_meter[k] += v.item() * batch_gpu["2d_bbx"].size(0)
            train_num_samples += batch_gpu["2d_bbx"].size(0)
        if rank == 0:
            iterator.set_postfix(loss=f"{total_loss.item():.4f}")
    
    try:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    except:
        pass
    # Train metric
    local_avg_metrics = {k: v / train_num_samples for k, v in train_meter.items()}
    train_avg_metrics = reduce_dict(local_avg_metrics, world_size, average=True)
    
    return train_avg_metrics

@torch.no_grad()
def evaluate(
    cfg : Any,
    model: torch.nn.Module,
    model_ema: torch.nn.Module,
    val_metric, ema_metric,
    rank: int,
    epoch: int,
    val_dataloader: Iterable,
    device: torch.device,
    criterion: torch.nn.Module,
    world_size: int,
    vis_dir: Path = None,
):
    model.eval()
    model_ema.module.eval()
    torch.cuda.empty_cache()
    val_metric.reset()
    ema_metric.reset()
    val_meter = defaultdict(float)
    ema_meter = defaultdict(float)
    val_num_samples = 0
    if rank == 0:
        print(f"Starting validation for Epoch {epoch+1}...")
    
    with torch.inference_mode():
        for i, val_batch in enumerate(val_dataloader):
            val_batch_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
            val_outputs = model(
                bbx2d_tight=val_batch_gpu["2d_bbx"],
                mask=val_batch_gpu["padding_mask"],
                # f_metric=val_batch_gpu["feat_metric"],
                # f_mono=val_batch_gpu["feat_mono"],
                f_dino=val_batch_gpu["feat_dino"],
            )
            ema_outputs = model_ema.module(
                bbx2d_tight=val_batch_gpu["2d_bbx"],
                mask=val_batch_gpu["padding_mask"],
                # f_metric=val_batch_gpu["feat_metric"],
                # f_mono=val_batch_gpu["feat_mono"],
                f_dino=val_batch_gpu["feat_dino"],
            )
            # 1. Calculate loss first (expects normalized [0,1] depth)
            val_loss_dict = criterion(val_outputs, val_batch_gpu)
            ema_loss_dict = criterion(ema_outputs, val_batch_gpu)

            # Legacy depth-map heads emit unconstrained map; scalar heads already output metric depth.
            
            val_metric.update(val_outputs, val_batch_gpu)
            ema_metric.update(ema_outputs, val_batch_gpu)
            for k in val_loss_dict.keys():
                if isinstance(val_loss_dict[k], torch.Tensor):
                    val_meter[k] += val_loss_dict[k].item() * val_batch_gpu["2d_bbx"].size(0)
                if isinstance(ema_loss_dict[k], torch.Tensor):
                    ema_meter[k] += ema_loss_dict[k].item() * val_batch_gpu["2d_bbx"].size(0)
            val_num_samples += val_batch_gpu["2d_bbx"].size(0)
            if rank ==0 and i == 0:
                _visualize_validataion(val_outputs, ema_outputs, val_batch_gpu, epoch, vis_dir)
                
    try:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    except:
        pass
    
    total_samples_tensor = torch.tensor([val_num_samples], device=device)
    torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
    total_val_samples = total_samples_tensor.item()
    
    local_sum_metrics = {k: v for k, v in val_meter.items()}
    local_ema_metrics = {k: v for k, v in ema_meter.items()}
    global_sum_metrics = reduce_dict(local_sum_metrics, world_size, average=False)
    global_ema_metrics = reduce_dict(local_ema_metrics, world_size, average=False)
    global_val_metrics = {k: v / total_val_samples for k, v in global_sum_metrics.items()}
    global_ema_metrics = {k: v / total_val_samples for k, v in global_ema_metrics.items()}
    global_val_dists = reduce_dict(val_metric.get_result_dict(), world_size, average=True)
    global_ema_dists = reduce_dict(ema_metric.get_result_dict(), world_size, average=True)
    
    return global_val_metrics, global_ema_metrics, global_val_dists, global_ema_dists
       
def _visualize_validataion(val_outputs, ema_outputs, val_batch, epoch, vis_dir):
    # Clone tensors to avoid CUDA Graphs issues with torch.compile
    h_maps = val_outputs['corner heatmaps'][0].clone()  # [8, 128, 128]
    p_coords_128 = val_outputs['corner coords'][0].clone() / 4.0
    g_coords_128 = val_batch['gt_corners'][0].clone() * 128.0
    
    h_ema_maps = ema_outputs['corner heatmaps'][0].clone()  # [8, 128, 128]
    p_ema_coords_128 = ema_outputs['corner coords'][0].clone() / 4.0
    g_ema_coords_128 = val_batch['gt_corners'][0].clone() * 128.0
    
    val_save_name = vis_dir / f"epoch_{epoch+1:03d}_val.png"
    ema_save_name = vis_dir / f"epoch_{epoch+1:03d}_ema.png"
    
    vis_dir.mkdir(parents=True, exist_ok=True)
    visualize_heatmaps(h_maps, p_coords_128, g_coords_128, val_save_name)
    visualize_heatmaps(h_ema_maps, p_ema_coords_128, g_ema_coords_128, ema_save_name)
