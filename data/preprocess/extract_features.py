import hashlib
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

MOCA_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MOCA_ROOT.parent
sys.path.insert(0, str(MOCA_ROOT))

from data.data_utils import filtered_annotations
from data.image_dataloader import _default_transform
from models.feature_modules import DINOv3FeatureExtractor


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


def _collect_json_paths(json_file, split: str):
    target_path = _resolve_path(json_file) if json_file is not None else MOCA_ROOT / "datasets" / "MoCA3D"
    if target_path is None or not target_path.exists():
        raise FileNotFoundError(f"JSON path not found: {target_path}")

    if target_path.is_file():
        if target_path.suffix != ".json":
            raise ValueError(f"Expected a json file, got: {target_path}")
        if not target_path.stem.endswith(f"_{split}"):
            raise ValueError(f"JSON file {target_path.name} does not match split={split}")
        return [target_path]

    json_paths = sorted(target_path.glob(f"*_{split}.json"))
    if not json_paths:
        raise ValueError(f"No json files found in {target_path} for split={split}")
    return json_paths


def _default_feature_root(cfg, split: str):
    data_root = _resolve_path(cfg.data_root)
    return data_root / "moca_features" / split


def _default_image_map_path(cfg, split: str):
    data_root = _resolve_path(cfg.data_root)
    return data_root / f"image_map_{split}.json"


def get_filtered_unique_images(json_paths, target_quality: str, dino_img_size: int, min_area: int):
    unique_paths = set()
    dataset_counts = {}

    for path in json_paths:
        filtered_data = filtered_annotations(
            path,
            target_quality=target_quality,
            dino_img_size=dino_img_size,
            min_area=min_area,
        )
        dataset_name = path.stem.rsplit("_", 1)[0]
        dataset_counts[dataset_name] = len(filtered_data["annotations"])
        valid_image_ids = {ann["image_id"] for ann in filtered_data["annotations"]}
        image_map = {img["id"]: img["file_path"] for img in filtered_data["images"]}
        for img_id in valid_image_ids:
            img_path = image_map.get(img_id)
            if img_path is not None:
                unique_paths.add(img_path)

    print(f"[ExtractFeatures] json files: {[path.name for path in json_paths]}")
    print(f"[ExtractFeatures] filtered annotation counts: {dataset_counts}")
    print(f"[ExtractFeatures] unique images: {len(unique_paths)}")
    return sorted(unique_paths)


def worker(rank, world_size, all_image_rel_paths, cfg, data_root, feature_root, image_map_parent):
    my_chunk = all_image_rel_paths[rank::world_size]
    device = torch.device(f"cuda:{rank}")
    dino_checkpoint = _resolve_path(cfg.dinov3_checkpoint_path)
    dino_extractor = DINOv3FeatureExtractor(checkpoint_path=dino_checkpoint, device=device)
    transform_dino = _default_transform(int(cfg.data.dino_image_size))

    local_mapping = {}
    progress = tqdm(my_chunk, desc=f"[GPU {rank}]", position=rank, leave=True)

    with torch.inference_mode():
        for rel_path in progress:
            full_path = data_root / rel_path
            if not full_path.exists():
                progress.write(f"Image not found: {full_path}")
                continue

            path_hash = hashlib.md5(str(rel_path).encode("utf-8")).hexdigest()
            feat_name = f"feat_{path_hash}.pth"
            feat_path = feature_root / feat_name
            if not feat_path.exists():
                with Image.open(full_path) as image:
                    image = image.convert("RGB")
                    image_dino = transform_dino(image).unsqueeze(0).to(device)
                feat_dino = dino_extractor(image_dino).to(torch.bfloat16).cpu()
                torch.save({"dino": feat_dino}, feat_path)
            local_mapping[str(rel_path)] = feat_name

    rank_map_path = image_map_parent / f"image_map_rank{rank}.json"
    with open(rank_map_path, "w") as handle:
        json.dump(local_mapping, handle, indent=2)


def extract_features_parallel(args):
    cfg = OmegaConf.load(_resolve_path(args.config))
    split = args.split
    json_paths = _collect_json_paths(args.json_file, split)

    feature_root = _resolve_path(args.feature_root) if args.feature_root is not None else _default_feature_root(cfg, split)
    image_map_path = _resolve_path(args.image_map_path) if args.image_map_path is not None else _default_image_map_path(cfg, split)
    data_root = _resolve_path(args.data_root) if args.data_root is not None else _resolve_path(cfg.data_root)

    feature_root.mkdir(parents=True, exist_ok=True)
    image_map_path.parent.mkdir(parents=True, exist_ok=True)

    target_quality = args.target_quality if args.target_quality is not None else str(cfg.data.target_quality)
    dino_img_size = int(args.dino_img_size if args.dino_img_size is not None else cfg.data.dino_image_size)
    min_area = int(args.min_area if args.min_area is not None else cfg.data.min_area_object)

    image_rel_paths = get_filtered_unique_images(
        json_paths=json_paths,
        target_quality=target_quality,
        dino_img_size=dino_img_size,
        min_area=min_area,
    )

    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found for feature extraction.")

    print(f"[ExtractFeatures] split={split} | world_size={world_size}")
    print(f"[ExtractFeatures] data_root={data_root}")
    print(f"[ExtractFeatures] feature_root={feature_root}")
    print(f"[ExtractFeatures] image_map_path={image_map_path}")

    mp.spawn(
        worker,
        args=(world_size, image_rel_paths, cfg, data_root, feature_root, image_map_path.parent),
        nprocs=world_size,
        join=True,
    )

    final_mapping = {}
    if image_map_path.exists():
        with open(image_map_path, "r") as handle:
            final_mapping = json.load(handle)

    for rank in range(world_size):
        rank_map_path = image_map_path.parent / f"image_map_rank{rank}.json"
        with open(rank_map_path, "r") as handle:
            final_mapping.update(json.load(handle))
        rank_map_path.unlink()

    with open(image_map_path, "w") as handle:
        json.dump(final_mapping, handle, indent=2)
    print(f"[ExtractFeatures] completed: {image_map_path}")


def parse_args():
    parser = ArgumentParser(description="Extract DINO features for MoCA3D images")
    parser.add_argument("--config", type=str, default=str(MOCA_ROOT / "configs" / "MoCA_config.yaml"))
    parser.add_argument("--json-file", type=str, default=str(MOCA_ROOT / "datasets" / "MoCA3D"))
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--feature-root", type=str, default=None)
    parser.add_argument("--image-map-path", type=str, default=None)
    parser.add_argument("--target-quality", type=str, default=None)
    parser.add_argument("--dino-img-size", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    extract_features_parallel(parse_args())
