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
from losses.bbx3d_loss import BBox3DLoss
from models.moca_3d import Moca3DModel
from models.moca_3d_cube import BBox3DMLP
from utils.functions import set_seed
from utils.iou3d import IoU3DComputer

DEFAULT_CUBE_CKPT_DIR = MOCA_ROOT / "checkpoints" / "MoCA3D_Cube"
BEST_JOINT_CKPT = "best_iou_inv_joint.pt"
BEST_MOCA_CKPT = "best_iou_inv_moca.pth"
BEST_CUBE_CKPT_NAMES = ("best_iou_inv.pth", "best_iou_inv_cube.pth")


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


def _extract_named_state_dict(checkpoint_obj, candidate_keys):
    if not isinstance(checkpoint_obj, dict):
        return None
    for key in candidate_keys:
        if key not in checkpoint_obj:
            continue
        value = checkpoint_obj[key]
        if isinstance(value, dict):
            return _strip_ckpt_prefixes(_extract_state_dict(value))
    return None


def _split_prefixed_joint_state_dict(checkpoint_obj):
    if not isinstance(checkpoint_obj, dict):
        return None, None

    source_state = None
    for key in ("state_dict", "model", "model_state_dict", "joint_state_dict", "joint_model"):
        if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
            source_state = checkpoint_obj[key]
            break

    if source_state is None:
        tensor_items = {key: value for key, value in checkpoint_obj.items() if isinstance(value, torch.Tensor)}
        if tensor_items:
            source_state = tensor_items

    if source_state is None:
        return None, None

    state_dict = _strip_ckpt_prefixes(source_state)
    moca_prefixes = (
        "moca_model.",
        "moca.",
        "model.moca_model.",
        "joint_model.moca_model.",
    )
    cube_prefixes = (
        "cube_model.",
        "bbx3d_model.",
        "bbox3d_model.",
        "cube_head.",
        "bbx3d_head.",
        "joint_model.cube_model.",
        "joint_model.bbx3d_model.",
    )

    moca_state_dict = {}
    cube_state_dict = {}

    for key, value in state_dict.items():
        for prefix in moca_prefixes:
            if key.startswith(prefix):
                moca_state_dict[key[len(prefix):]] = value
                break
        else:
            for prefix in cube_prefixes:
                if key.startswith(prefix):
                    cube_state_dict[key[len(prefix):]] = value
                    break

    if not moca_state_dict:
        moca_state_dict = None
    if not cube_state_dict:
        cube_state_dict = None
    return moca_state_dict, cube_state_dict


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


def _clean_iou_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor[~torch.isnan(tensor)]
    return tensor.clamp(min=0.0, max=1.0)


def _safe_mean(tensor: torch.Tensor) -> float:
    return float(tensor.mean().item()) if tensor.numel() > 0 else 0.0


def _safe_max(tensor: torch.Tensor) -> float:
    return float(tensor.max().item()) if tensor.numel() > 0 else 0.0


def _infer_available_datasets(moca_cfg, args):
    if args.loader == "image":
        root = _resolve_path(moca_cfg.json_root)
        available_names = sorted({path.stem.rsplit("_", 1)[0] for path in root.glob("*_test.json")})
    else:
        root = _resolve_path(args.wds_root if args.wds_root is not None else moca_cfg.wds_root)
        available_names = sorted({path.name.rsplit("_", 1)[0] for path in root.glob("*_test") if path.is_dir()})

    if not available_names:
        raise ValueError(f"No datasets found for loader={args.loader}, split=test")

    if args.datasets is None:
        return available_names

    missing = [name for name in args.datasets if name not in available_names]
    if missing:
        raise ValueError(
            f"Requested datasets not found for loader={args.loader}, split=test: {missing}. "
            f"Available: {available_names}"
        )
    return args.datasets


def _resolve_eval_checkpoint_paths(args):
    checkpoint_dir = _resolve_path(args.checkpoint_dir)
    if checkpoint_dir is None or not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    joint_path = checkpoint_dir / BEST_JOINT_CKPT
    if joint_path.exists():
        return {
            "mode": "joint",
            "joint_path": joint_path,
            "moca_path": joint_path,
            "cube_path": joint_path,
        }

    moca_path = checkpoint_dir / BEST_MOCA_CKPT
    if not moca_path.exists():
        raise FileNotFoundError(f"MoCA checkpoint not found: {moca_path}")

    cube_path = None
    for cube_name in BEST_CUBE_CKPT_NAMES:
        candidate = checkpoint_dir / cube_name
        if candidate.exists():
            cube_path = candidate
            break
    if cube_path is None:
        expected = ", ".join(BEST_CUBE_CKPT_NAMES)
        raise FileNotFoundError(f"Cube checkpoint not found in {checkpoint_dir}. Expected one of: {expected}")

    return {
        "mode": "separate",
        "joint_path": None,
        "moca_path": moca_path,
        "cube_path": cube_path,
    }


def _load_eval_states(cfg, checkpoint_paths, device):
    if checkpoint_paths["mode"] == "joint":
        checkpoint_obj = _load_checkpoint(checkpoint_paths["joint_path"], map_location=device)
        moca_state_dict = _extract_named_state_dict(
            checkpoint_obj,
            ("moca_model", "moca_state_dict", "moca_model_ema", "moca"),
        )
        cube_state_dict = _extract_named_state_dict(
            checkpoint_obj,
            ("cube_model", "bbx3d_model", "bbox3d_model", "cube_model_ema", "bbx3d_model_ema", "cube"),
        )
        if moca_state_dict is None or cube_state_dict is None:
            split_moca_state_dict, split_cube_state_dict = _split_prefixed_joint_state_dict(checkpoint_obj)
            if moca_state_dict is None:
                moca_state_dict = split_moca_state_dict
            if cube_state_dict is None:
                cube_state_dict = split_cube_state_dict
        if moca_state_dict is None or cube_state_dict is None:
            available_keys = sorted(checkpoint_obj.keys()) if isinstance(checkpoint_obj, dict) else []
            raise KeyError(
                "Could not resolve MoCA/Cube states from joint checkpoint. "
                f"Available top-level keys: {available_keys}"
            )
        return {
            "moca_state_dict": moca_state_dict,
            "cube_state_dict": cube_state_dict,
            "moca_label": f"{checkpoint_paths['joint_path']} (moca_model)",
            "cube_label": f"{checkpoint_paths['joint_path']} (cube_model)",
        }

    cube_checkpoint_obj = _load_checkpoint(checkpoint_paths["cube_path"], map_location=device)
    if isinstance(cube_checkpoint_obj, dict) and "cube_model" in cube_checkpoint_obj:
        cube_state_dict = _strip_ckpt_prefixes(cube_checkpoint_obj["cube_model"])
    else:
        cube_state_dict = _strip_ckpt_prefixes(_extract_state_dict(cube_checkpoint_obj))

    moca_checkpoint_obj = _load_checkpoint(checkpoint_paths["moca_path"], map_location=device)
    return {
        "moca_state_dict": _strip_ckpt_prefixes(_extract_state_dict(moca_checkpoint_obj)),
        "cube_state_dict": cube_state_dict,
        "moca_label": str(checkpoint_paths["moca_path"]),
        "cube_label": str(checkpoint_paths["cube_path"]),
    }


def _build_image_loader(moca_cfg, args, dataset_name: str, rank: int, world_size: int):
    return build_image_dataloader(
        root_dir=_resolve_path(moca_cfg.json_root),
        data_dir=_resolve_path(moca_cfg.data_root),
        seed=int(moca_cfg.get("seed", 42)),
        split="test",
        batch_size=int(moca_cfg.batch_size),
        dino_image_size=int(moca_cfg.data.dino_image_size),
        target_quality=str(moca_cfg.data.target_quality),
        min_area=int(moca_cfg.data.min_area_object),
        shuffle=False,
        num_workers=int(moca_cfg.num_workers),
        datasets=[dataset_name],
        epoch_length=None,
        is_ddp=(world_size > 1),
        rank=rank,
        world_size=world_size,
    )


def _build_wds_loader(moca_cfg, args, dataset_name: str, world_size: int):
    wds_root = _resolve_path(args.wds_root if args.wds_root is not None else moca_cfg.wds_root)
    return build_wds_feature_dataloader(
        cfg=moca_cfg,
        wds_root=wds_root,
        split="test",
        enable_aug=False,
        dino_img_size=int(moca_cfg.data.dino_image_size),
        epoch_length=0,
        world_size=world_size,
        include_datasets=[dataset_name],
        full_dataset_eval=True,
        random_mix=False,
        max_cap=-1,
    )[0]


def _build_loader(moca_cfg, args, dataset_name: str, rank: int, world_size: int):
    if args.loader == "image":
        return _build_image_loader(moca_cfg, args, dataset_name, rank, world_size)
    if args.loader == "wds":
        return _build_wds_loader(moca_cfg, args, dataset_name, world_size)
    raise ValueError(f"Unsupported loader type: {args.loader}")


def _forward_moca(moca_model, batch, loader_name: str):
    if loader_name == "wds":
        moca_model.feature_mode = True
        return moca_model(
            f_dino=batch["feat_dino"],
            bbx2d_tight=batch["2d_bbx"],
            mask=batch["padding_mask"],
            return_decoder_feat=True,
        )

    moca_model.feature_mode = False
    return moca_model(
        images_dino=batch["image_dino"],
        bbx2d_tight=batch["2d_bbx"],
        mask=batch["padding_mask"],
        return_decoder_feat=True,
    )


def _move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def _evaluate_single_dataset(rank, device, moca_model, cube_model, criterion, dataloader, loader_name: str, dataset_name: str):
    iou_computer = IoU3DComputer(device=device)
    iou_list = []
    iou_inv_list = []

    moca_model.eval()
    cube_model.eval()

    iterator = dataloader
    if rank == 0:
        iterator = tqdm(dataloader, desc=f"Eval [{dataset_name}]", leave=True)

    for batch in iterator:
        batch_gpu = _move_batch_to_device(batch, device)
        moca_outputs = _forward_moca(moca_model, batch_gpu, loader_name)
        cube_outputs = cube_model(moca_outputs, batch_gpu["K"])
        pred_boxes = criterion.build_bbox3d(
            cube_outputs["centers"],
            cube_outputs["sizes"],
            cube_outputs["yaws"],
            cube_outputs["ray_x"],
            cube_outputs["ray_z"],
            ray_y=cube_outputs.get("ray_y", None),
        )

        gt_boxes = batch_gpu["3d_bbx"]
        if isinstance(gt_boxes, list):
            gt_boxes = torch.stack(
                [torch.as_tensor(x, dtype=torch.float32, device=device) for x in gt_boxes],
                dim=0,
            )

        iou = iou_computer.iou3d_pytorch3d_safe(pred_boxes, gt_boxes, verbose=False)
        iou_inv = iou_computer.invariant_iou3d(pred_boxes, gt_boxes)
        iou_list.append(iou.diag().clamp(min=0.0, max=1.0))
        iou_inv_list.append(iou_inv.diag().clamp(min=0.0, max=1.0))

    local_iou = torch.cat(iou_list, dim=0) if iou_list else torch.zeros(0, device=device)
    local_iou_inv = torch.cat(iou_inv_list, dim=0) if iou_inv_list else torch.zeros(0, device=device)

    return {
        "dataset": dataset_name,
        "iou": _gather_variable_length_tensor(local_iou),
        "iou_inv": _gather_variable_length_tensor(local_iou_inv),
    }


def _print_iou_metrics(title: str, rank: int, all_iou: torch.Tensor, all_iou_inv: torch.Tensor):
    if rank != 0:
        return

    all_iou = _clean_iou_tensor(all_iou)
    all_iou_inv = _clean_iou_tensor(all_iou_inv)
    num_samples = int(all_iou_inv.numel())
    if num_samples <= 0:
        print(f"{title}: no valid samples")
        return

    zero_ratio = float((all_iou_inv == 0).sum().item()) / max(num_samples, 1)
    print("\n" + "=" * 60)
    print(title)
    print("-" * 60)
    print(f"Samples: {num_samples}")
    print(f"IoU=0 ratio (invariant): {zero_ratio:.4f}")
    print(f"[Naive]     mean={_safe_mean(all_iou):.4f} | max={_safe_max(all_iou):.4f}")
    print(f"[Invariant] mean={_safe_mean(all_iou_inv):.4f} | max={_safe_max(all_iou_inv):.4f}")
    print("=" * 60)


def evaluate_worker(rank: int, world_size: int, args) -> None:
    try:
        _setup_dist(rank, world_size, args)
        device = torch.device(f"cuda:{rank}")

        cfg = OmegaConf.load(_resolve_path(args.config))
        set_seed(int(cfg.get("seed", 42)), rank)

        moca_cfg = OmegaConf.load(_resolve_path(cfg.get("moca_config_path", "configs/MoCA_config.yaml")))
        moca_cfg.batch_size = int(cfg.batch_size)
        moca_cfg.num_workers = int(cfg.num_workers)
        moca_cfg.device = str(device)
        moca_cfg.data.dino_image_size = int(cfg.get("dino_img_size", moca_cfg.data.dino_image_size))
        moca_cfg.feature_mode = args.loader == "wds"
        if args.wds_root is not None:
            moca_cfg.wds_root = args.wds_root

        datasets = _infer_available_datasets(moca_cfg, args)
        checkpoint_paths = _resolve_eval_checkpoint_paths(args)
        state_pack = _load_eval_states(cfg, checkpoint_paths, device)

        moca_model = Moca3DModel(moca_cfg).to(device)
        moca_incompat = moca_model.load_state_dict(state_pack["moca_state_dict"], strict=False)
        cube_model = BBox3DMLP(hidden_dim=int(cfg.hidden_dim)).to(device)
        cube_model.load_state_dict(state_pack["cube_state_dict"], strict=True)
        criterion = BBox3DLoss().to(device)

        if rank == 0:
            print("=" * 72)
            print(f"Evaluate cube head | loader={args.loader} | world_size={world_size}")
            print(f"checkpoint_dir: {_resolve_path(args.checkpoint_dir)}")
            print(f"checkpoint_mode: {checkpoint_paths['mode']}")
            print(f"MoCA state: {state_pack['moca_label']}")
            print(f"Cube state: {state_pack['cube_label']}")
            print(f"datasets: {datasets}")
            print(
                f"MoCA load(strict=False): missing={len(moca_incompat.missing_keys)} "
                f"unexpected={len(moca_incompat.unexpected_keys)}"
            )
            print("=" * 72)

        results = []
        for dataset_name in datasets:
            dataloader = _build_loader(moca_cfg, args, dataset_name, rank, world_size)
            result = _evaluate_single_dataset(
                rank=rank,
                device=device,
                moca_model=moca_model,
                cube_model=cube_model,
                criterion=criterion,
                dataloader=dataloader,
                loader_name=args.loader,
                dataset_name=dataset_name,
            )
            results.append(result)
            _print_iou_metrics(f"Dataset: {dataset_name}", rank, result["iou"], result["iou_inv"])

        if len(results) > 1:
            merged_iou = torch.cat([result["iou"] for result in results], dim=0) if results else torch.zeros(0, device=device)
            merged_iou_inv = torch.cat([result["iou_inv"] for result in results], dim=0) if results else torch.zeros(0, device=device)
            _print_iou_metrics("Dataset: TOTAL", rank, merged_iou, merged_iou_inv)
    except Exception as exc:
        import traceback

        print(f"Exception in rank {rank}: {exc}")
        traceback.print_exc()
        raise
    finally:
        _cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MoCA cube checkpoints with IoU3D metrics")
    parser.add_argument("--config", type=str, default=str(MOCA_ROOT / "configs" / "MoCA_cube_config.yaml"))
    parser.add_argument("--loader", choices=["image", "wds"], default="wds")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CUBE_CKPT_DIR))
    parser.add_argument("--wds-root", type=str, default=None)
    parser.add_argument("--master-addr", type=str, default="localhost")
    parser.add_argument("--master-port", type=int, default=12355)
    return parser.parse_args()


def main():
    args = parse_args()
    args.datasets = _normalize_dataset_args(args.datasets)

    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found.")

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    mp.set_sharing_strategy("file_system")
    mp.spawn(evaluate_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
