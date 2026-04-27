import argparse
import os
import signal
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm

signal.signal(signal.SIGHUP, signal.SIG_IGN)

MOCA_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MOCA_ROOT.parent
sys.path.insert(0, str(MOCA_ROOT))

from data.image_dataloader import build_image_dataloader
from data.wds_dataloader import build_wds_feature_dataloader
from models.moca_3d import Moca3DModel
from utils.functions import set_seed
from utils.iou3d import IoU3DComputer
from utils.nhd import NHDComputer
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
        PROJECT_ROOT / path_obj,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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


def _all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _gather_variable_length_tensor(local_tensor: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return local_tensor

    world_size = dist.get_world_size()
    local_len = torch.tensor([local_tensor.numel()], device=local_tensor.device, dtype=torch.long)
    gathered_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(gathered_lens, local_len)
    gathered_lens = torch.cat(gathered_lens, dim=0)
    max_len = int(gathered_lens.max().item()) if gathered_lens.numel() > 0 else 0
    if max_len <= 0:
        return torch.zeros(0, device=local_tensor.device, dtype=local_tensor.dtype)

    if local_tensor.numel() < max_len:
        pad = torch.full(
            (max_len - local_tensor.numel(),),
            float("nan"),
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
        local_tensor = torch.cat([local_tensor, pad], dim=0)

    gathered = [torch.empty(max_len, device=local_tensor.device, dtype=local_tensor.dtype) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)

    chunks = []
    for idx, length in enumerate(gathered_lens.detach().cpu().tolist()):
        length = int(length)
        if length > 0:
            chunks.append(gathered[idx][:length])
    if not chunks:
        return torch.zeros(0, device=local_tensor.device, dtype=local_tensor.dtype)
    return torch.cat(chunks, dim=0)


def _clean_metric_tensor(tensor: torch.Tensor, clamp_max: float | None = None) -> torch.Tensor:
    if clamp_max is not None:
        tensor = torch.clamp(tensor, max=clamp_max)
    return tensor[~torch.isnan(tensor)]


def _safe_mean(tensor: torch.Tensor) -> float:
    return float(tensor.mean().item()) if tensor.numel() > 0 else 0.0


def _safe_median(tensor: torch.Tensor) -> float:
    return float(tensor.median().item()) if tensor.numel() > 0 else 0.0


def _safe_max(tensor: torch.Tensor) -> float:
    return float(tensor.max().item()) if tensor.numel() > 0 else 0.0


def _infer_available_datasets(cfg, args):
    if args.datasets is not None:
        return args.datasets

    split = args.split
    if args.loader == "image":
        root = _resolve_path(cfg.json_root)
        names = sorted({path.stem.rsplit("_", 1)[0] for path in root.glob(f"*_{split}.json")})
    else:
        root = _resolve_path(cfg.wds_root)
        names = sorted({path.name.rsplit("_", 1)[0] for path in root.glob(f"*_{split}") if path.is_dir()})

    if not names:
        raise ValueError(f"No datasets found for loader={args.loader}, split={split}")
    return names


def _build_image_loader(cfg, args, dataset_name: str, rank: int, world_size: int):
    return build_image_dataloader(
        root_dir=_resolve_path(cfg.json_root),
        data_dir=_resolve_path(cfg.data_root),
        seed=int(cfg.get("seed", 42)),
        split=args.split,
        batch_size=int(cfg.batch_size),
        dino_image_size=int(cfg.data.dino_image_size),
        target_quality=str(cfg.data.target_quality),
        min_area=int(cfg.data.min_area_object),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        datasets=[dataset_name],
        epoch_length=None,
        is_ddp=(world_size > 1),
        rank=rank,
        world_size=world_size,
    )


def _build_wds_loader(cfg, args, dataset_name: str, world_size: int):
    original_workers = cfg.num_workers
    cfg.num_workers = int(cfg.num_workers)
    try:
        return build_wds_feature_dataloader(
            cfg=cfg,
            wds_root=_resolve_path(cfg.wds_root),
            split=args.split,
            enable_aug=False,
            dino_img_size=int(cfg.data.dino_image_size),
            epoch_length=0,
            world_size=world_size,
            include_datasets=[dataset_name],
            full_dataset_eval=True,
            random_mix=False,
            max_cap=-1,
        )[0]
    finally:
        cfg.num_workers = original_workers


def _build_loader(cfg, args, dataset_name: str, rank: int, world_size: int):
    if args.loader == "image":
        return _build_image_loader(cfg, args, dataset_name, rank, world_size)
    if args.loader == "wds":
        return _build_wds_loader(cfg, args, dataset_name, world_size)
    raise ValueError(f"Unsupported loader type: {args.loader}")


def _forward_model(model, batch, loader_name: str):
    if loader_name == "wds":
        model.feature_mode = True
        return model(
            f_dino=batch["feat_dino"],
            bbx2d_tight=batch["2d_bbx"],
            mask=batch["padding_mask"],
        )

    model.feature_mode = False
    return model(
        images_dino=batch["image_dino"],
        bbx2d_tight=batch["2d_bbx"],
        mask=batch["padding_mask"],
    )


def _compute_metrics_for_dataset(model, device, dataloader, loader_name: str, rank: int, dataset_name: str, rectify_mode: str):
    metric = CornerGeometryMetric(device=device, use_hungarian_matching=True)
    iou_computer = IoU3DComputer(device=device, rectify_mode=rectify_mode)
    nhd_computer = NHDComputer(device=device)

    model.eval()
    iterator = dataloader
    if rank == 0:
        iterator = tqdm(dataloader, desc=f"Evaluate [{dataset_name}]", leave=True)

    with torch.inference_mode():
        for batch in iterator:
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = _forward_model(model, batch, loader_name)
            metric.update(outputs, batch)
            iou_computer.update(outputs, batch)
            nhd_computer.update(outputs, batch)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    result = {
        "uv_sum": _all_reduce_sum(metric.total_uv_dist.clone()),
        "depth_sum": _all_reduce_sum(metric.total_depth_diff.clone()),
        "depth_diff_rate": _all_reduce_sum(metric.total_depth_diff_rate.clone()),
        "samples": _all_reduce_sum(metric.total_samples.clone()),
        "ious": _gather_variable_length_tensor(
            torch.cat(iou_computer.iou_list, dim=0) if len(iou_computer.iou_list) > 0 else torch.zeros(0, device=device)
        ),
        "ious_inv": _gather_variable_length_tensor(
            torch.cat(iou_computer.iou_list_convex, dim=0) if len(iou_computer.iou_list_convex) > 0 else torch.zeros(0, device=device)
        ),
        "nhd": _gather_variable_length_tensor(
            torch.cat(nhd_computer.nhd_list, dim=0) if len(nhd_computer.nhd_list) > 0 else torch.zeros(0, device=device)
        ),
    }
    return result


def _print_result(title: str, result: dict, rank: int) -> None:
    if rank != 0:
        return

    total_samples = int(result["samples"].item())
    total_corners = max(total_samples * 8, 1)
    thresholds = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    ious = _clean_metric_tensor(result["ious"], clamp_max=1.0)
    ious_inv = _clean_metric_tensor(result["ious_inv"], clamp_max=1.0)
    nhd = _clean_metric_tensor(result["nhd"])

    print("\n" + "=" * 60)
    print(title)
    print("-" * 60)
    print(f"Samples: {total_samples}")
    print(
        f"Corner Geometry | avg_uv_error={(result['uv_sum'] / total_corners).item():.4f} px "
        f"| avg_depth_error={(result['depth_sum'] / total_corners).item():.4f} m "
        f"| avg_depth_diff_rate={(result['depth_diff_rate'] / total_corners * 100.0).item():.2f}%"
    )
    print(
        f"IoU3D          | standard_avg={_safe_mean(ious):.4f} | standard_max={_safe_max(ious):.4f} "
        f"| invariant_avg={_safe_mean(ious_inv):.4f} | invariant_max={_safe_max(ious_inv):.4f}"
    )
    print(
        f"NHD            | avg={_safe_mean(nhd):.4f} | median={_safe_median(nhd):.4f} | max={_safe_max(nhd):.4f}"
    )
    for threshold in thresholds:
        ratio = float((nhd <= threshold).sum().item()) / max(total_samples, 1)
        print(f"NHD <= {threshold}: {ratio:.4f}")
    if ious_inv.numel() > 0:
        print(f"IoU=0 ratio    | {float((ious_inv == 0).sum().item()) / max(int((ious_inv <= 1.0).sum().item()), 1):.4f}")
    print("=" * 60)


def _merge_results(results: list[dict], device: torch.device) -> dict:
    if not results:
        return {
            "uv_sum": torch.tensor(0.0, device=device),
            "depth_sum": torch.tensor(0.0, device=device),
            "depth_diff_rate": torch.tensor(0.0, device=device),
            "samples": torch.tensor(0, device=device, dtype=torch.long),
            "ious": torch.zeros(0, device=device),
            "ious_inv": torch.zeros(0, device=device),
            "nhd": torch.zeros(0, device=device),
        }

    return {
        "uv_sum": sum(result["uv_sum"] for result in results),
        "depth_sum": sum(result["depth_sum"] for result in results),
        "depth_diff_rate": sum(result["depth_diff_rate"] for result in results),
        "samples": sum(result["samples"] for result in results),
        "ious": torch.cat([result["ious"] for result in results], dim=0),
        "ious_inv": torch.cat([result["ious_inv"] for result in results], dim=0),
        "nhd": torch.cat([result["nhd"] for result in results], dim=0),
    }


def evaluate_worker(rank: int, world_size: int, cfg, args) -> None:
    try:
        _setup_dist(rank, world_size, args)
        set_seed(int(cfg.get("seed", 42)), rank)
        device = torch.device(f"cuda:{rank}")
        cfg.device = str(device)
        cfg.feature_mode = args.loader == "wds"

        checkpoint_path = _resolve_path(args.checkpoint)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        datasets = _infer_available_datasets(cfg, args)

        if rank == 0:
            print("=" * 80)
            print(f"loader={args.loader} | split={args.split} | datasets={datasets}")
            print(f"checkpoint={checkpoint_path}")
            print("=" * 80)

        model = Moca3DModel(cfg).to(device)
        checkpoint_obj = _load_checkpoint(checkpoint_path, map_location=device)
        state_dict = _strip_ckpt_prefixes(_extract_state_dict(checkpoint_obj))
        model.load_state_dict(state_dict, strict=True)

        if rank == 0:
            print(f"\nEvaluating checkpoint: {checkpoint_path}")

        dataset_results = []
        for dataset_name in datasets:
            dataloader = _build_loader(cfg, args, dataset_name, rank, world_size)
            result = _compute_metrics_for_dataset(
                model=model,
                device=device,
                dataloader=dataloader,
                loader_name=args.loader,
                rank=rank,
                dataset_name=dataset_name,
                rectify_mode=args.rectify_mode,
            )
            dataset_results.append(result)
            _print_result(f"Checkpoint={checkpoint_path.name} | Dataset={dataset_name}", result, rank)

        if len(dataset_results) > 1:
            merged = _merge_results(dataset_results, device)
            _print_result(f"Checkpoint={checkpoint_path.name} | Dataset=TOTAL", merged, rank)

    except Exception as exc:
        import traceback

        print(f"Exception in rank {rank}: {exc}")
        traceback.print_exc()
    finally:
        _cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone evaluation entrypoint for MoCA3D")
    parser.add_argument("--config", type=str, default=str(MOCA_ROOT / "configs" / "MoCA_config.yaml"))
    parser.add_argument("--loader", choices=["image", "wds"], default="image")
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--checkpoint", type=str, default=str(MOCA_ROOT / "checkpoints" / "moca3d.safetensors"))
    parser.add_argument("--rectify-mode", choices=["pca", "kabsch"], default="kabsch")

    parser.add_argument("--master-addr", type=str, default="localhost")
    parser.add_argument("--master-port", type=int, default=12355)
    return parser.parse_args()


def main():
    args = parse_args()
    args.datasets = _normalize_dataset_args(args.datasets)
    cfg = OmegaConf.load(_resolve_path(args.config))

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available.")

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    mp.set_sharing_strategy("file_system")
    mp.spawn(evaluate_worker, args=(world_size, cfg, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
