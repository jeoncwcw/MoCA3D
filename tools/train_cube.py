import argparse
import math
import os
import signal
import sys
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from timm.utils import ModelEmaV2
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

signal.signal(signal.SIGHUP, signal.SIG_IGN)

try:
    import torch._functorch.config

    torch._functorch.config.donated_buffer = False
except Exception:
    pass

MOCA_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MOCA_ROOT.parent
sys.path.insert(0, str(MOCA_ROOT))

from data.image_dataloader import build_image_dataloader
from data.wds_dataloader import build_wds_feature_dataloader
from losses.bbx3d_loss import BBox3DLoss, prepare_gt_for_loss
from models.moca_3d import Moca3DModel
from models.moca_3d_cube import BBox3DMLP
from utils.functions import reduce_dict, set_seed
from utils.iou3d import IoU3DComputer


def _resolve_path(path_like):
    if path_like is None:
        return None
    path_obj = Path(path_like).expanduser()
    if path_obj.is_absolute():
        return path_obj

    candidates = [
        Path.cwd() / path_obj,
        MOCA_ROOT / path_obj,
        PROJECT_ROOT / path_obj,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _normalize_dataset_args(values):
    if values is None:
        return None
    normalized = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip().strip("[]").strip().strip("'\"")
            if token:
                normalized.append(token)
    if not normalized:
        return None
    seen = set()
    deduped = []
    for value in normalized:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model"):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]
        return checkpoint_obj
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint_obj)}")


def _load_checkpoint(checkpoint_path: Path, map_location):
    if checkpoint_path.suffix == ".safetensors":
        return load_safetensors(str(checkpoint_path), device=str(map_location))
    return torch.load(checkpoint_path, map_location=map_location)


def _strip_ckpt_prefixes(state_dict):
    return {
        key.replace("module.", "").replace("_orig_mod.", ""): value
        for key, value in state_dict.items()
    }


def _unwrap_model(model):
    actual = model
    while True:
        if hasattr(actual, "module"):
            actual = actual.module
            continue
        if hasattr(actual, "_orig_mod"):
            actual = actual._orig_mod
            continue
        return actual


def _setup_dist(rank: int, world_size: int, args) -> None:
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.set_device(rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=rank)
    except Exception:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cudnn.rnn.fp32_precision = "tf32"
    torch.set_float32_matmul_precision("high")


def _cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _build_scheduler(optimizer, num_epochs, warmup_epochs, steps_per_epoch):
    total_steps = max(num_epochs * steps_per_epoch, 1)
    warmup_steps = max(warmup_epochs * steps_per_epoch, 1)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _forward_moca(moca_model, batch_gpu, loader_name: str):
    actual_model = _unwrap_model(moca_model)
    if loader_name == "wds":
        actual_model.feature_mode = True
        return moca_model(
            f_dino=batch_gpu["feat_dino"],
            bbx2d_tight=batch_gpu["2d_bbx"],
            mask=batch_gpu["padding_mask"],
            return_decoder_feat=True,
        )

    actual_model.feature_mode = False
    return moca_model(
        images_dino=batch_gpu["image_dino"],
        bbx2d_tight=batch_gpu["2d_bbx"],
        mask=batch_gpu["padding_mask"],
        return_decoder_feat=True,
    )


def _save_joint_checkpoint(
    save_path,
    epoch,
    moca_model,
    cube_model,
    optimizer,
    scheduler,
    scaler,
    best_iou_inv,
    moca_ema=None,
    cube_ema=None,
):
    payload = {
        "epoch": epoch,
        "moca_model": _unwrap_model(moca_model).state_dict(),
        "cube_model": _unwrap_model(cube_model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_iou_inv": best_iou_inv,
    }
    if moca_ema is not None:
        payload["moca_model_ema"] = _unwrap_model(moca_ema.module).state_dict()
    if cube_ema is not None:
        payload["cube_model_ema"] = _unwrap_model(cube_ema.module).state_dict()
    torch.save(payload, save_path)


def _load_resume_checkpoint(
    resume_path,
    moca_model,
    cube_model,
    optimizer,
    scheduler,
    scaler,
    device,
    moca_ema=None,
    cube_ema=None,
):
    ckpt = torch.load(resume_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported resume checkpoint format: {type(ckpt)}")

    if "moca_model" in ckpt:
        _unwrap_model(moca_model).load_state_dict(_strip_ckpt_prefixes(ckpt["moca_model"]), strict=True)
    if "cube_model" in ckpt:
        _unwrap_model(cube_model).load_state_dict(_strip_ckpt_prefixes(ckpt["cube_model"]), strict=True)
    if moca_ema is not None and "moca_model_ema" in ckpt:
        _unwrap_model(moca_ema.module).load_state_dict(_strip_ckpt_prefixes(ckpt["moca_model_ema"]), strict=True)
    if cube_ema is not None and "cube_model_ema" in ckpt:
        _unwrap_model(cube_ema.module).load_state_dict(_strip_ckpt_prefixes(ckpt["cube_model_ema"]), strict=True)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_iou_inv = float(ckpt.get("best_iou_inv", 0.0))
    return start_epoch, best_iou_inv


def _get_epoch_length(cfg, args, split: str) -> int:
    if split == "train" and args.train_epoch_length is not None:
        return int(args.train_epoch_length)
    if split != "train" and args.val_epoch_length is not None:
        return int(args.val_epoch_length)
    return int(cfg.get("train_epoch_length")) if split == "train" else int(cfg.get("val_epoch_length"))


def _build_image_loader(moca_cfg, cfg, args, split: str, include_datasets, rank: int, world_size: int):
    epoch_length = _get_epoch_length(cfg, args, split)
    return build_image_dataloader(
        root_dir=_resolve_path(moca_cfg.json_root),
        data_dir=_resolve_path(moca_cfg.data_root),
        seed=int(cfg.get("seed", 42)),
        split=split,
        batch_size=int(cfg.batch_size),
        dino_image_size=int(moca_cfg.data.dino_image_size),
        target_quality=str(moca_cfg.data.target_quality),
        min_area=int(moca_cfg.data.min_area_object),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        datasets=include_datasets,
        epoch_length=epoch_length,
        is_ddp=(world_size > 1),
        rank=rank,
        world_size=world_size,
    )


def _build_wds_loader(moca_cfg, cfg, args, split: str, include_datasets, world_size: int):
    return build_wds_feature_dataloader(
        cfg=moca_cfg,
        wds_root=_resolve_path(cfg.wds_root),
        split=split,
        enable_aug=False,
        dino_img_size=int(cfg.get("dino_img_size", moca_cfg.data.dino_image_size)),
        epoch_length=_get_epoch_length(cfg, args, split),
        world_size=world_size,
        include_datasets=include_datasets,
        full_dataset_eval=False,
        random_mix=bool(cfg.get("random_mix", True)) if split == "train" else False,
        max_cap=int(cfg.get("max_cap", 4)) if split == "train" else -1,
    )


def _build_loader(moca_cfg, cfg, args, loader_name: str, split: str, include_datasets, rank: int, world_size: int):
    if loader_name == "image":
        return _build_image_loader(moca_cfg, cfg, args, split, include_datasets, rank, world_size), None
    if loader_name == "wds":
        return _build_wds_loader(moca_cfg, cfg, args, split, include_datasets, world_size)
    raise ValueError(f"Unsupported loader type: {loader_name}")


def train_one_epoch(
    rank,
    world_size,
    moca_model,
    cube_model,
    moca_ema,
    cube_ema,
    criterion,
    train_loader,
    train_loader_name,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    num_batches_per_epoch,
    grad_accum_steps,
    grad_clip_norm,
    amp_enabled,
):
    moca_model.train()
    cube_model.train()
    train_meter = defaultdict(float)
    train_num_samples = 0
    num_seen_batches = 0

    optimizer.zero_grad(set_to_none=True)
    iterator = train_loader
    if rank == 0:
        iterator = tqdm(train_loader, desc=f"Epoch [{epoch + 1}] Training", leave=True, total=num_batches_per_epoch)

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for step, batch in enumerate(iterator):
        batch_gpu = _move_batch_to_device(batch, device)
        gt_3d_bbox = prepare_gt_for_loss(batch_gpu["3d_bbx"])

        moca_outputs = _forward_moca(moca_model, batch_gpu, train_loader_name)
        with torch.amp.autocast(autocast_device, enabled=amp_enabled):
            cube_outputs = cube_model(moca_outputs, batch_gpu["K"])
            loss_dict = criterion(cube_outputs, gt_3d_bbox)
            total_loss = loss_dict["total_loss"] / grad_accum_steps

        scaler.scale(total_loss).backward()
        do_step = (step + 1) % grad_accum_steps == 0
        if do_step:
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    chain(moca_model.parameters(), cube_model.parameters()),
                    max_norm=grad_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            moca_ema.update(moca_model)
            cube_ema.update(cube_model)
            scheduler.step()

        batch_size = batch_gpu["2d_bbx"].size(0)
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                train_meter[key] += value.item() * batch_size
        train_num_samples += batch_size
        num_seen_batches += 1
        if rank == 0:
            iterator.set_postfix(loss=f"{loss_dict['total_loss'].item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    if num_seen_batches > 0 and (num_seen_batches % grad_accum_steps) != 0:
        scaler.unscale_(optimizer)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                chain(moca_model.parameters(), cube_model.parameters()),
                max_norm=grad_clip_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        moca_ema.update(moca_model)
        cube_ema.update(cube_model)
        scheduler.step()

    if dist.is_initialized():
        dist.barrier()

    local_avg_metrics = {key: value / train_num_samples for key, value in train_meter.items()} if train_num_samples > 0 else {"total_loss": float("nan")}
    return reduce_dict(local_avg_metrics, world_size, average=True)


@torch.no_grad()
def evaluate(
    rank,
    world_size,
    moca_model,
    cube_model,
    moca_ema,
    cube_ema,
    criterion,
    val_loader,
    val_loader_name,
    device,
    epoch,
):
    moca_model.eval()
    cube_model.eval()
    moca_ema.module.eval()
    cube_ema.module.eval()
    val_meter = defaultdict(float)
    ema_meter = defaultdict(float)
    val_num_samples = 0

    iou_computer = IoU3DComputer(device=device)
    iou_computer_ema = IoU3DComputer(device=device)
    iou_sum = torch.zeros(1, device=device)
    iou_inv_sum = torch.zeros(1, device=device)
    iou_sum_ema = torch.zeros(1, device=device)
    iou_inv_sum_ema = torch.zeros(1, device=device)
    iou_count = 0

    if rank == 0:
        print(f"Starting validation for Epoch {epoch + 1}...")

    with torch.inference_mode():
        for batch in val_loader:
            batch_gpu = _move_batch_to_device(batch, device)
            gt_3d_bbox = prepare_gt_for_loss(batch_gpu["3d_bbx"])

            moca_outputs = _forward_moca(moca_model, batch_gpu, val_loader_name)
            cube_outputs = cube_model(moca_outputs, batch_gpu["K"])
            loss_dict = criterion(cube_outputs, gt_3d_bbox)

            moca_outputs_ema = _forward_moca(moca_ema.module, batch_gpu, val_loader_name)
            cube_outputs_ema = cube_ema.module(moca_outputs_ema, batch_gpu["K"])
            loss_dict_ema = criterion(cube_outputs_ema, gt_3d_bbox)

            pred_boxes = criterion.build_bbox3d(
                cube_outputs["centers"],
                cube_outputs["sizes"],
                cube_outputs["yaws"],
                cube_outputs["ray_x"],
                cube_outputs["ray_z"],
                ray_y=cube_outputs.get("ray_y", None),
            )
            pred_boxes_ema = criterion.build_bbox3d(
                cube_outputs_ema["centers"],
                cube_outputs_ema["sizes"],
                cube_outputs_ema["yaws"],
                cube_outputs_ema["ray_x"],
                cube_outputs_ema["ray_z"],
                ray_y=cube_outputs_ema.get("ray_y", None),
            )

            iou = iou_computer.iou3d_pytorch3d_safe(pred_boxes, batch_gpu["3d_bbx"], verbose=False)
            iou_inv = iou_computer.invariant_iou3d(pred_boxes, batch_gpu["3d_bbx"])
            iou_ema = iou_computer_ema.iou3d_pytorch3d_safe(pred_boxes_ema, batch_gpu["3d_bbx"], verbose=False)
            iou_inv_ema = iou_computer_ema.invariant_iou3d(pred_boxes_ema, batch_gpu["3d_bbx"])

            diag_iou = iou.diag().clamp(min=0.0, max=1.0)
            diag_iou_inv = iou_inv.diag().clamp(min=0.0, max=1.0)
            diag_iou_ema = iou_ema.diag().clamp(min=0.0, max=1.0)
            diag_iou_inv_ema = iou_inv_ema.diag().clamp(min=0.0, max=1.0)
            iou_sum += diag_iou.sum()
            iou_inv_sum += diag_iou_inv.sum()
            iou_sum_ema += diag_iou_ema.sum()
            iou_inv_sum_ema += diag_iou_inv_ema.sum()
            iou_count += int(diag_iou.numel())

            batch_size = batch_gpu["2d_bbx"].size(0)
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    val_meter[key] += value.item() * batch_size
                if isinstance(loss_dict_ema[key], torch.Tensor):
                    ema_meter[key] += loss_dict_ema[key].item() * batch_size
            val_num_samples += batch_size

    if dist.is_initialized():
        dist.barrier()

    total_samples_tensor = torch.tensor([float(val_num_samples)], device=device)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    total_val_samples = total_samples_tensor.item()

    global_sum_metrics = reduce_dict(dict(val_meter), world_size, average=False)
    global_sum_metrics_ema = reduce_dict(dict(ema_meter), world_size, average=False)
    if total_val_samples > 0:
        global_val_metrics = {key: value / total_val_samples for key, value in global_sum_metrics.items()}
        global_ema_metrics = {key: value / total_val_samples for key, value in global_sum_metrics_ema.items()}
    else:
        global_val_metrics = {"total_loss": float("nan")}
        global_ema_metrics = {"total_loss": float("nan")}

    iou_stats = torch.tensor(
        [iou_sum.item(), iou_inv_sum.item(), iou_sum_ema.item(), iou_inv_sum_ema.item(), float(iou_count)],
        device=device,
    )
    dist.all_reduce(iou_stats, op=dist.ReduceOp.SUM)
    if iou_stats[4].item() > 0:
        global_val_metrics["iou3d"] = (iou_stats[0] / iou_stats[4]).item()
        global_val_metrics["iou3d_invariant"] = (iou_stats[1] / iou_stats[4]).item()
        global_ema_metrics["iou3d"] = (iou_stats[2] / iou_stats[4]).item()
        global_ema_metrics["iou3d_invariant"] = (iou_stats[3] / iou_stats[4]).item()
    else:
        global_val_metrics["iou3d"] = 0.0
        global_val_metrics["iou3d_invariant"] = 0.0
        global_ema_metrics["iou3d"] = 0.0
        global_ema_metrics["iou3d_invariant"] = 0.0
    return global_val_metrics, global_ema_metrics


def train_worker(rank, world_size, args):
    try:
        _setup_dist(rank, world_size, args)
        device = torch.device(f"cuda:{rank}")

        cfg = OmegaConf.load(_resolve_path(args.config))
        train_include_datasets = _normalize_dataset_args(args.train_datasets)
        val_include_datasets = _normalize_dataset_args(args.val_datasets)
        if val_include_datasets is None:
            val_include_datasets = train_include_datasets
        set_seed(int(cfg.get("seed", 42)), rank)

        amp_enabled = bool(cfg.get("amp", True)) and device.type == "cuda"
        moca_cfg = OmegaConf.load(_resolve_path(cfg.get("moca_config_path", "configs/MoCA_config.yaml")))
        moca_cfg.device = str(device)
        moca_cfg.batch_size = int(cfg.batch_size)
        moca_cfg.num_workers = int(cfg.num_workers)
        moca_cfg.data.dino_image_size = int(cfg.get("dino_img_size", moca_cfg.data.dino_image_size))

        need_image_backbone = (args.train_loader == "image") or (args.val_loader == "image")
        moca_cfg.feature_mode = not need_image_backbone

        moca_model = Moca3DModel(moca_cfg).to(device)
        moca_ckpt_path = _resolve_path(cfg.get("moca_checkpoint_path", "checkpoints/moca3d.safetensors"))
        if not moca_ckpt_path.exists():
            raise FileNotFoundError(f"MoCA checkpoint not found: {moca_ckpt_path}")
        moca_ckpt = _load_checkpoint(moca_ckpt_path, map_location=device)
        moca_state_dict = _strip_ckpt_prefixes(_extract_state_dict(moca_ckpt))
        strict_load = bool(cfg.get("moca_checkpoint_strict", False))
        incompat = moca_model.load_state_dict(moca_state_dict, strict=strict_load)
        if rank == 0:
            print(
                f"Loaded MoCA checkpoint: {moca_ckpt_path} "
                f"(strict={strict_load}, missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)})"
            )

        moca_model = DDP(moca_model, device_ids=[rank], find_unused_parameters=False, gradient_as_bucket_view=True)
        cube_model = BBox3DMLP(hidden_dim=int(cfg.hidden_dim)).to(device)
        cube_model = DDP(cube_model, device_ids=[rank], find_unused_parameters=False, gradient_as_bucket_view=True)
        if args.compile and hasattr(torch, "compile"):
            moca_model = torch.compile(moca_model)
            cube_model = torch.compile(cube_model)

        ema_decay = float(cfg.get("ema_decay", 0.999))
        moca_ema = ModelEmaV2(moca_model, decay=ema_decay, device=device)
        cube_ema = ModelEmaV2(cube_model, decay=ema_decay, device=device)
        criterion = BBox3DLoss().to(device)

        cube_lr = float(cfg.learning_rate)
        moca_lr_ratio = float(cfg.get("moca_lr_ratio", 0.02))
        moca_lr = float(cfg.get("moca_learning_rate", cube_lr * moca_lr_ratio))
        moca_weight_decay = float(cfg.get("moca_weight_decay", cfg.weight_decay))
        optimizer = optim.AdamW(
            [
                {"params": moca_model.parameters(), "lr": moca_lr, "weight_decay": moca_weight_decay},
                {"params": cube_model.parameters(), "lr": cube_lr, "weight_decay": float(cfg.weight_decay)},
            ],
            lr=cube_lr,
            weight_decay=float(cfg.weight_decay),
        )

        train_loader, aug_obj = _build_loader(
            moca_cfg=moca_cfg,
            cfg=cfg,
            args=args,
            loader_name=args.train_loader,
            split="train",
            include_datasets=train_include_datasets,
            rank=rank,
            world_size=world_size,
        )
        val_loader, _ = _build_loader(
            moca_cfg=moca_cfg,
            cfg=cfg,
            args=args,
            loader_name=args.val_loader,
            split="val",
            include_datasets=val_include_datasets,
            rank=rank,
            world_size=world_size,
        )

        train_num_batches = max(_get_epoch_length(cfg, args, "train") // (int(cfg.batch_size) * world_size), 1)
        scheduler = _build_scheduler(
            optimizer=optimizer,
            num_epochs=int(cfg.num_epochs),
            warmup_epochs=int(cfg.get("warmup_epochs", 5)),
            steps_per_epoch=train_num_batches,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        checkpoint_dir = MOCA_ROOT / "checkpoints" / str(cfg.model_name)
        if rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        start_epoch = 0
        best_iou_inv = float(cfg.get("best_iou_inv", 0.0))
        resume_path = cfg.get("resume_checkpoint_path", None)
        if resume_path:
            resume_path = _resolve_path(resume_path)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            start_epoch, best_iou_inv = _load_resume_checkpoint(
                resume_path=resume_path,
                moca_model=moca_model,
                cube_model=cube_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                moca_ema=moca_ema,
                cube_ema=cube_ema,
            )
            if rank == 0:
                print(f"Resumed from {resume_path} at epoch {start_epoch} | best_iou_inv={best_iou_inv:.4f}")

        num_epochs = int(cfg.num_epochs)
        grad_accum_steps = max(int(cfg.get("grad_accum_steps", 1)), 1)
        grad_clip_norm = float(cfg.get("grad_clip_norm", 5.0))
        val_interval = int(cfg.get("val_interval", 1))
        save_interval = int(cfg.get("save_interval", 5))

        if rank == 0:
            print("=" * 72)
            print(f"Start cube joint FT | model={cfg.model_name} | world_size={world_size}")
            print(f"MoCA checkpoint: {moca_ckpt_path}")
            print(f"loaders: train={args.train_loader}, val={args.val_loader}")
            print(f"datasets: train={train_include_datasets}, val={val_include_datasets}")
            print(f"LR (MoCA / Cube): {moca_lr:.2e} / {cube_lr:.2e}")
            print(f"checkpoints -> {checkpoint_dir}")
            print("=" * 72)

        start_time = time.time()
        for epoch in range(start_epoch, num_epochs):
            train_sampler = getattr(train_loader, "sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            val_sampler = getattr(val_loader, "sampler", None)
            if hasattr(val_sampler, "set_epoch"):
                val_sampler.set_epoch(epoch)
            if aug_obj is not None and hasattr(aug_obj, "set_epoch"):
                aug_obj.set_epoch(epoch)

            train_metrics = train_one_epoch(
                rank=rank,
                world_size=world_size,
                moca_model=moca_model,
                cube_model=cube_model,
                moca_ema=moca_ema,
                cube_ema=cube_ema,
                criterion=criterion,
                train_loader=train_loader,
                train_loader_name=args.train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                epoch=epoch,
                num_batches_per_epoch=train_num_batches,
                grad_accum_steps=grad_accum_steps,
                grad_clip_norm=grad_clip_norm,
                amp_enabled=amp_enabled,
            )

            val_metrics = None
            ema_metrics = None
            if ((epoch + 1) % val_interval == 0) or (epoch > num_epochs // 2):
                val_metrics, ema_metrics = evaluate(
                    rank=rank,
                    world_size=world_size,
                    moca_model=moca_model,
                    cube_model=cube_model,
                    moca_ema=moca_ema,
                    cube_ema=cube_ema,
                    criterion=criterion,
                    val_loader=val_loader,
                    val_loader_name=args.val_loader,
                    device=device,
                    epoch=epoch,
                )
                if rank == 0:
                    curr_iou_inv = float(val_metrics.get("iou3d_invariant", 0.0))
                    curr_iou_inv_ema = float(ema_metrics.get("iou3d_invariant", 0.0))
                    use_ema_ckpt = curr_iou_inv_ema >= curr_iou_inv
                    best_curr = max(curr_iou_inv, curr_iou_inv_ema)
                    best_joint_path = checkpoint_dir / "best_iou_inv_joint.pt"
                    if best_curr > best_iou_inv or not best_joint_path.exists():
                        best_iou_inv = best_curr
                        best_source_moca = moca_ema.module if use_ema_ckpt else moca_model
                        best_source_cube = cube_ema.module if use_ema_ckpt else cube_model
                        torch.save(_unwrap_model(best_source_moca).state_dict(), checkpoint_dir / "best_iou_inv_moca.pth")
                        torch.save(_unwrap_model(best_source_cube).state_dict(), checkpoint_dir / "best_iou_inv_cube.pth")
                        _save_joint_checkpoint(
                            save_path=best_joint_path,
                            epoch=epoch,
                            moca_model=best_source_moca,
                            cube_model=best_source_cube,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            best_iou_inv=best_iou_inv,
                            moca_ema=moca_ema,
                            cube_ema=cube_ema,
                        )
                        source = "EMA" if use_ema_ckpt else "Online"
                        print(
                            f"Saved best cube FT checkpoint ({source}) | epoch={epoch + 1} "
                            f"| val_iou_inv={curr_iou_inv:.4f} | ema_iou_inv={curr_iou_inv_ema:.4f}"
                        )

            if rank == 0 and (epoch + 1) % save_interval == 0:
                _save_joint_checkpoint(
                    save_path=checkpoint_dir / f"epoch{epoch + 1:03d}.pt",
                    epoch=epoch,
                    moca_model=moca_model,
                    cube_model=cube_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_iou_inv=best_iou_inv,
                    moca_ema=moca_ema,
                    cube_ema=cube_ema,
                )
                print(f"Saved periodic checkpoint at epoch {epoch + 1}")

            if rank == 0:
                _save_joint_checkpoint(
                    save_path=checkpoint_dir / "latest.pt",
                    epoch=epoch,
                    moca_model=moca_model,
                    cube_model=cube_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_iou_inv=best_iou_inv,
                    moca_ema=moca_ema,
                    cube_ema=cube_ema,
                )
                elapsed = time.time() - start_time
                lrs = optimizer.param_groups
                train_msg = (
                    f"Epoch [{epoch + 1:03d}/{num_epochs:03d}] "
                    f"train_total={train_metrics.get('total_loss', float('nan')):.6f} "
                    f"train_all={train_metrics.get('loss_all', float('nan')):.6f} "
                    f"train_center={train_metrics.get('loss_center', float('nan')):.6f} "
                    f"train_size={train_metrics.get('loss_size', float('nan')):.6f} "
                    f"train_yaw={train_metrics.get('loss_yaw', float('nan')):.6f} "
                    f"lr_moca={lrs[0]['lr']:.2e} lr_cube={lrs[1]['lr']:.2e}"
                )
                if val_metrics is not None:
                    val_msg = (
                        f" | val_total={val_metrics.get('total_loss', float('nan')):.6f} "
                        f"val_iou={val_metrics.get('iou3d', 0.0):.4f} "
                        f"val_iou_inv={val_metrics.get('iou3d_invariant', 0.0):.4f}"
                    )
                    ema_msg = (
                        f" | ema_total={ema_metrics.get('total_loss', float('nan')):.6f} "
                        f"ema_iou={ema_metrics.get('iou3d', 0.0):.4f} "
                        f"ema_iou_inv={ema_metrics.get('iou3d_invariant', 0.0):.4f}"
                    )
                else:
                    val_msg = ""
                    ema_msg = ""
                print(train_msg + val_msg + ema_msg + f" | best_iou_inv={best_iou_inv:.4f} | elapsed={elapsed:.1f}s")

            if dist.is_initialized():
                dist.barrier()

        if rank == 0:
            print("=" * 72)
            print(f"Training complete | best invariant IoU: {best_iou_inv:.4f}")
            print("=" * 72)
    except Exception as exc:
        import traceback

        print(f"Exception in rank {rank}: {exc}")
        traceback.print_exc()
        raise
    finally:
        _cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Train MoCA cube head with joint MoCA fine-tuning")
    parser.add_argument("--config", type=str, default=str(MOCA_ROOT / "configs" / "MoCA_cube_config.yaml"))
    parser.add_argument("--train-loader", choices=["image", "wds"], default="wds")
    parser.add_argument("--val-loader", choices=["image", "wds"], default="wds")
    parser.add_argument("--train-datasets", nargs="*", default=None)
    parser.add_argument("--val-datasets", nargs="*", default=None)
    parser.add_argument("--train-epoch-length", type=int, default=None)
    parser.add_argument("--val-epoch-length", type=int, default=None)
    parser.add_argument("--compile", dest="compile", action="store_true")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.set_defaults(compile=True)
    parser.add_argument("--allow-existing-checkpoint-dir", action="store_true")
    parser.add_argument("--master-addr", type=str, default="localhost")
    parser.add_argument("--master-port", type=int, default=12355)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(_resolve_path(args.config))
    checkpoint_dir = MOCA_ROOT / "checkpoints" / str(cfg.model_name)
    if checkpoint_dir.exists() and not args.allow_existing_checkpoint_dir:
        raise RuntimeError(
            f"Checkpoint directory already exists: {checkpoint_dir}. "
            f"Use --allow-existing-checkpoint-dir to reuse it."
        )
    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found.")

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    mp.set_sharing_strategy("file_system")
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
