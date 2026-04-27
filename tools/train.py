import argparse
import os
import signal
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from timm.utils import ModelEmaV2
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

signal.signal(signal.SIGHUP, signal.SIG_IGN)

MOCA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOCA_ROOT))

from data.image_dataloader import build_image_dataloader
from data.wds_dataloader import build_wds_feature_dataloader
from losses.criterion import BETRLoss
from models.moca_3d import Moca3DModel
from utils.engine import evaluate, train_one_epoch
from utils.functions import get_parameter_groups, get_scheduler, print_epoch_stats, set_seed
from utils.projective_fidelity import CornerGeometryMetric


def _resolve_path(path_like):
    if path_like is None:
        return None
    path_obj = Path(path_like).expanduser()
    if path_obj.is_absolute():
        return path_obj

    candidates = [
        Path.cwd() / path_obj,
        MOCA_ROOT / path_obj,
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


class LoaderModeModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loader_mode = "image"

    def set_loader_mode(self, loader_mode: str) -> None:
        self.loader_mode = loader_mode

    def forward(self, bbx2d_tight, mask=None, images_dino=None, f_dino=None, **kwargs):
        if self.loader_mode == "wds":
            self.model.feature_mode = True
            if f_dino is None:
                f_dino = images_dino
            kwargs.pop("images_dino", None)
            return self.model(bbx2d_tight=bbx2d_tight, mask=mask, f_dino=f_dino, **kwargs)

        self.model.feature_mode = False
        if images_dino is None:
            images_dino = f_dino
        kwargs.pop("f_dino", None)
        return self.model(bbx2d_tight=bbx2d_tight, mask=mask, images_dino=images_dino, **kwargs)


class AliasLoader:
    def __init__(self, loader, source_key: str, target_key: str):
        self.loader = loader
        self.source_key = source_key
        self.target_key = target_key
        self.sampler = getattr(loader, "sampler", None)
        self.dataset = getattr(loader, "dataset", None)

    def __iter__(self):
        for batch in self.loader:
            if self.target_key in batch or self.source_key not in batch:
                yield batch
                continue
            batch = dict(batch)
            batch[self.target_key] = batch[self.source_key]
            yield batch

    def __len__(self):
        return len(self.loader)


def _maybe_alias_loader(loader, loader_name: str, phase: str):
    if phase == "train" and loader_name == "wds":
        return AliasLoader(loader, source_key="feat_dino", target_key="image_dino")
    if phase == "val" and loader_name == "image":
        return AliasLoader(loader, source_key="image_dino", target_key="feat_dino")
    return loader


def _unwrap_for_loader_mode(model):
    actual = model
    while hasattr(actual, "module"):
        actual = actual.module
    return actual


def _unwrap_for_state_dict(model):
    actual = model
    while True:
        if hasattr(actual, "module"):
            actual = actual.module
            continue
        if hasattr(actual, "model"):
            actual = actual.model
            continue
        return actual


def _set_loader_mode(model, loader_mode: str) -> None:
    actual = _unwrap_for_loader_mode(model)
    if hasattr(actual, "set_loader_mode"):
        actual.set_loader_mode(loader_mode)


def _get_state_dict(model):
    return _unwrap_for_state_dict(model).state_dict()


def _setup_ddp(rank: int, world_size: int, args) -> None:
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


def _checkpoint_dir(cfg, args) -> Path:
    return MOCA_ROOT / "checkpoints" / str(cfg.model_name)


def _vis_dir(cfg, args) -> Path:
    if args.vis_dir is not None:
        return _resolve_path(args.vis_dir)
    return MOCA_ROOT / "outputs" / str(cfg.model_name) / "heatmap_vis"


def _override_cfg(cfg, args) -> None:
    scalar_overrides = {
        "init_checkpoint_path": args.init_checkpoint,
    }
    
    for key, value in scalar_overrides.items():
        if value is not None:
            cfg[key] = value
    if args.train_epoch_length is not None:
        cfg.epoch_length = args.train_epoch_length


def _build_image_loader(cfg, args, split: str, rank: int, world_size: int):
    dataset_filter = args.train_datasets if split == "train" else args.val_datasets
    epoch_length = args.train_epoch_length if split == "train" else args.val_epoch_length
    if epoch_length is None and split == "train":
        epoch_length = int(cfg.get("epoch_length", 0)) or None

    return build_image_dataloader(
        root_dir=_resolve_path(cfg.json_root),
        data_dir=_resolve_path(cfg.data_root),
        seed=int(cfg.get("seed", 42)),
        split=split,
        batch_size=int(cfg.batch_size),
        dino_image_size=int(cfg.data.dino_image_size),
        target_quality=str(cfg.data.target_quality),
        min_area=int(cfg.data.min_area_object),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        datasets=dataset_filter,
        epoch_length=epoch_length,
        is_ddp=(world_size > 1),
        rank=rank,
        world_size=world_size,
    ), None


def _build_wds_loader(cfg, args, split: str, world_size: int):
    dataset_filter = args.train_datasets if split == "train" else args.val_datasets
    epoch_length = args.train_epoch_length if split == "train" else args.val_epoch_length
    if epoch_length is None:
        epoch_length = int(cfg.get("epoch_length", 50000))


    return build_wds_feature_dataloader(
        cfg=cfg,
        wds_root=_resolve_path(cfg.wds_root),
        split=split,
        dino_img_size=int(cfg.data.dino_image_size),
        epoch_length=epoch_length,
        world_size=world_size,
        include_datasets=dataset_filter,
        random_mix=True,
        max_cap=(args.max_cap if split == "train" else -1),
    )


def _build_loader(cfg, args, loader_name: str, split: str, rank: int, world_size: int):
    if loader_name == "image":
        return _build_image_loader(cfg, args, split, rank, world_size)
    if loader_name == "wds":
        return _build_wds_loader(cfg, args, split, world_size)
    raise ValueError(f"Unsupported loader type: {loader_name}")


def _num_batches_per_epoch(train_loader, cfg, args, world_size: int) -> int:
    try:
        return len(train_loader)
    except TypeError:
        epoch_length = args.train_epoch_length or int(cfg.get("epoch_length", 0))
        if epoch_length <= 0:
            raise ValueError("Unable to infer num_batches_per_epoch; set --train-epoch-length.")
        return max(1, epoch_length // (int(cfg.batch_size) * world_size))


def train_worker(rank: int, world_size: int, cfg, args) -> None:
    try:
        import torch._functorch.config

        torch._functorch.config.donated_buffer = False
        _setup_ddp(rank, world_size, args)
        set_seed(int(cfg.get("seed", 42)), rank)

        device = torch.device(f"cuda:{rank}")
        need_image_backbone = args.train_loader == "image" or args.val_loader == "image"
        cfg.device = str(device)
        cfg.feature_mode = not need_image_backbone

        base_model = Moca3DModel(cfg).to(device)
        init_checkpoint = cfg.get("init_checkpoint_path", None)
        if init_checkpoint:
            init_checkpoint_path = _resolve_path(init_checkpoint)
            if not init_checkpoint_path.exists():
                raise FileNotFoundError(f"Initial checkpoint not found: {init_checkpoint_path}")
            checkpoint_obj = _load_checkpoint(init_checkpoint_path, map_location="cpu")
            state_dict = _strip_ckpt_prefixes(_extract_state_dict(checkpoint_obj))
            strict_load = bool(cfg.get("init_checkpoint_strict", True))
            if args.init_checkpoint_strict is not None:
                strict_load = args.init_checkpoint_strict
            incompat = base_model.load_state_dict(state_dict, strict=strict_load)
            if rank == 0:
                print(
                    f"Loaded init checkpoint: {init_checkpoint_path} "
                    f"(strict={strict_load}, missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)})"
                )

        model = LoaderModeModel(base_model)
        if args.compile and hasattr(torch, "compile"):
            model = torch.compile(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False, gradient_as_bucket_view=True)
        model_ema = ModelEmaV2(model, decay=args.ema_decay, device=device)

        criterion = BETRLoss(cfg).to(device)
        lr_multiplier = {
            "transformer_decoder": cfg.lr_multipliers.decoder,
            "box_embedding": cfg.lr_multipliers.box_embedding,
        }
        param_groups = get_parameter_groups(model, weight_decay=cfg.weight_decay, lr_multiplier=lr_multiplier)
        for group in param_groups:
            group["lr"] = cfg.learning_rate * group.pop("lr_scale", 1.0)

        optimizer = optim.AdamW(
            param_groups,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        train_loader_raw, aug_obj = _build_loader(cfg, args, args.train_loader, "train", rank, world_size)
        train_loader = _maybe_alias_loader(train_loader_raw, args.train_loader, "train")

  
        val_loader_raw, _ = _build_loader(cfg, args, args.val_loader, "val", rank, world_size)
        val_loader = _maybe_alias_loader(val_loader_raw, args.val_loader, "val")

        num_batches_per_epoch = _num_batches_per_epoch(train_loader, cfg, args, world_size)
        scheduler = get_scheduler(optimizer, cfg, num_batches_per_epoch)

        checkpoint_dir = _checkpoint_dir(cfg, args)
        vis_dir = _vis_dir(cfg, args)
        best_uv = float(cfg.get("best_uv", float("inf")))
        best_depth_rate = float(cfg.get("best_depth_rate", float("inf")))

        if rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            vis_dir.mkdir(parents=True, exist_ok=True)
            print("=" * 80)
            print(f"train_loader={args.train_loader} | val_loader={args.val_loader}")
            print(f"checkpoint_dir={checkpoint_dir}")
            print(f"vis_dir={vis_dir}")
            print("=" * 80)

        scaler = torch.amp.GradScaler("cuda")
        val_metric = CornerGeometryMetric(device=device)
        ema_metric = CornerGeometryMetric(device=device)
        num_epochs = int(cfg.num_epochs)

        for epoch in range(num_epochs):
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            if aug_obj is not None and hasattr(aug_obj, "set_epoch"):
                aug_obj.set_epoch(epoch)

            _set_loader_mode(model, args.train_loader)
            _set_loader_mode(model_ema.module, args.train_loader)
            train_metrics = train_one_epoch(
                model=model,
                model_ema=model_ema,
                train_dataloader=train_loader,
                rank=rank,
                epoch=epoch,
                num_batches_per_epoch=num_batches_per_epoch,
                device=device,
                criterion=criterion,
                grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                world_size=world_size,
            )

            global_val_loss = None
            global_ema_loss = None
            global_val_dists = None
            global_ema_dists = None
            if ((epoch) % int(cfg.get("val_interval", 1)) == 0) or (epoch > num_epochs // 2):
                val_sampler = getattr(val_loader, "sampler", None)
                if hasattr(val_sampler, "set_epoch"):
                    val_sampler.set_epoch(epoch)
                _set_loader_mode(model, args.val_loader)
                _set_loader_mode(model_ema.module, args.val_loader)
                global_val_loss, global_ema_loss, global_val_dists, global_ema_dists = evaluate(
                    cfg=cfg,
                    model=model,
                    model_ema=model_ema,
                    val_metric=val_metric,
                    ema_metric=ema_metric,
                    rank=rank,
                    epoch=epoch,
                    val_dataloader=val_loader,
                    device=device,
                    criterion=criterion,
                    world_size=world_size,
                    vis_dir=vis_dir,
                )

            if rank == 0:
                print_epoch_stats(epoch, num_epochs, train_metrics, global_val_loss, global_ema_loss)
                if global_val_dists is not None and global_ema_dists is not None:
                    print("---- Validation Corner Geometry Metrics ----")
                    print(
                        f"Standard Model - Average UV Error: {global_val_dists['avg_uv_error']:.4f} px, "
                        f"Average Depth Error: {global_val_dists['avg_depth_error']:.4f} meters"
                    )
                    print(
                        f"EMA Model      - Average UV Error: {global_ema_dists['avg_uv_error']:.4f} px, "
                        f"Average Depth Error: {global_ema_dists['avg_depth_error']:.4f} meters"
                    )
                    print("--------------------------------------------")

                    min_uv = min(global_val_dists["avg_uv_error"], global_ema_dists["avg_uv_error"])
                    min_depth_rate = min(
                        global_val_dists["avg_depth_diff_rate"],
                        global_ema_dists["avg_depth_diff_rate"],
                    )
                    if min_uv < best_uv:
                        best_uv = min_uv
                        best_uv_path = checkpoint_dir / "best_uv.pth"
                        source_model = model_ema.module if global_ema_dists["avg_uv_error"] < global_val_dists["avg_uv_error"] else model
                        torch.save(_get_state_dict(source_model), best_uv_path)
                        print(f"Saved best uv model to {best_uv_path}")
                    if min_depth_rate < best_depth_rate:
                        best_depth_rate = min_depth_rate
                        best_depth_path = checkpoint_dir / "best_depth_rate.pth"
                        source_model = (
                            model_ema.module
                            if global_ema_dists["avg_depth_diff_rate"] < global_val_dists["avg_depth_diff_rate"]
                            else model
                        )
                        torch.save(_get_state_dict(source_model), best_depth_path)
                        print(f"Saved best depth-rate model to {best_depth_path}")

                if (epoch + 1) % int(cfg.get("save_interval", 5)) == 0:
                    checkpoint_path = checkpoint_dir / f"epoch{epoch + 1:03d}.pth"
                    torch.save(_get_state_dict(model), checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

    except Exception as exc:
        import traceback

        print(f"Exception in rank {rank}: {exc}")
        traceback.print_exc()
    finally:
        _cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone trainer for MoCA3D")
    parser.add_argument("--config", type=str, default=str(MOCA_ROOT / "configs" / "MoCA_config.yaml"))
    parser.add_argument("--train-loader", choices=["image", "wds"], default="image")
    parser.add_argument("--val-loader", choices=["image", "wds"], default="wds")

    parser.add_argument("--vis-dir", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.set_defaults(init_checkpoint_strict=None)
    parser.add_argument("--init-checkpoint-strict", dest="init_checkpoint_strict", action="store_true")
    parser.add_argument("--no-init-checkpoint-strict", dest="init_checkpoint_strict", action="store_false")

    parser.add_argument("--train-datasets", nargs="*", default=None)
    parser.add_argument("--val-datasets", nargs="*", default=None)
    parser.add_argument("--train-epoch-length", type=int, default=None)
    parser.add_argument("--val-epoch-length", type=int, default=None)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-cap", type=int, default=4)

    parser.set_defaults(compile=True)
    parser.add_argument("--compile", dest="compile", action="store_true")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--allow-existing-checkpoint-dir", action="store_true")
    parser.add_argument("--master-addr", type=str, default="localhost")
    parser.add_argument("--master-port", type=int, default=12355)
    return parser.parse_args()


def main():
    args = parse_args()
    args.train_datasets = _normalize_dataset_args(args.train_datasets)
    args.val_datasets = _normalize_dataset_args(args.val_datasets)
    cfg = OmegaConf.load(_resolve_path(args.config))
    _override_cfg(cfg, args)

    checkpoint_dir = _checkpoint_dir(cfg, args)
    if checkpoint_dir.exists() and not args.allow_existing_checkpoint_dir:
        print(f"Checkpoint directory already exists: {checkpoint_dir}")
        print("Use --allow-existing-checkpoint-dir to reuse it.")
        return

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available.")

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    mp.set_sharing_strategy("file_system")
    mp.spawn(train_worker, args=(world_size, cfg, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
