import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from omegaconf import OmegaConf
from tqdm import tqdm

MOCA_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MOCA_ROOT.parent
sys.path.insert(0, str(MOCA_ROOT))

from data.data_utils import filtered_annotations

def process_sample(ann_data, pad_info, img_info, dino_img_size=512):
    pad_top, pad_left, new_h, new_w, scale = pad_info
    
    bbx2d_orig = np.array(ann_data["bbox2D_tight"], dtype=np.float32)
    bbx2d_processed = bbx2d_orig * scale
    bbx2d_processed[[0, 2]] += pad_left
    bbx2d_processed[[1, 3]] += pad_top
    bbx2d_processed = torch.tensor(bbx2d_processed, dtype=torch.float32) / dino_img_size

    corners_list = ann_data["projected_corners"]
    coords = [(float(c["u"]), float(c["v"])) for c in corners_list]
    gt_corners = torch.tensor(coords, dtype=torch.float32)
    
    gt_corners = gt_corners * scale
    gt_corners[:, 0] += pad_left
    gt_corners[:, 1] += pad_top
    gt_corners = gt_corners / dino_img_size # Normalize to [0,1] range

    # gt 3D box
    bbx_3d = ann_data["bbox3D_cam"]
    
    # depths processing
    raw_depths = torch.clamp(torch.tensor(ann_data["depth"], dtype=torch.float32), min=1e-3)
    # Apply Virtual Depth Normalization (Z_virt = Z / f)
    fy = img_info["K"][1][1]
    h = img_info["height"]
    f_v = h_v = 512.0
    virtual_scale = (f_v / fy) * (h / h_v)
    raw_depths = raw_depths * virtual_scale

    return {
        "2d_bbx": bbx2d_processed,
        "3d_bbx": bbx_3d,
        "gt_corners": gt_corners,
        "gt_depths": raw_depths
    }


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
    return _resolve_path(cfg.data_root) / "moca_features" / split


def _default_image_map_path(cfg, split: str):
    return _resolve_path(cfg.data_root) / f"image_map_{split}.json"


def main(args):
    cfg = OmegaConf.load(_resolve_path(args.config))
    split = args.split
    json_paths = _collect_json_paths(args.json_file, split)
    feature_root = _resolve_path(args.feature_root) if args.feature_root is not None else _default_feature_root(cfg, split)
    output_root = _resolve_path(args.output_root) if args.output_root is not None else _resolve_path(cfg.wds_root)
    image_map_path = _resolve_path(args.image_map_path) if args.image_map_path is not None else _default_image_map_path(cfg, split)
    target_quality = args.target_quality if args.target_quality is not None else str(cfg.data.target_quality)
    dino_img_size = int(args.dino_img_size if args.dino_img_size is not None else cfg.data.dino_image_size)
    min_area = int(args.min_area if args.min_area is not None else cfg.data.min_area_object)

    output_root.mkdir(parents=True, exist_ok=True)
    if not image_map_path.exists():
        raise FileNotFoundError(f"Image map not found: {image_map_path}")

    with open(image_map_path, "r") as handle:
        image_map = json.load(handle)

    print(f"[ConvertToWDS] split={split}")
    print(f"[ConvertToWDS] json files: {[path.name for path in json_paths]}")
    print(f"[ConvertToWDS] feature_root={feature_root}")
    print(f"[ConvertToWDS] output_root={output_root}")
    print(f"[ConvertToWDS] image_map_path={image_map_path}")

    for json_path in json_paths:
        dataset_name = json_path.stem.replace(f"_{split}", "")
        print(f"[ConvertToWDS] processing dataset: {dataset_name} ({split})")
        data = filtered_annotations(
            json_path,
            target_quality=target_quality,
            dino_img_size=dino_img_size,
            min_area=min_area,
        )
        img2anns = defaultdict(list)
        for ann in data["annotations"]:
            img2anns[ann["image_id"]].append(ann)
        img_info_map = {img["id"]: img for img in data["images"]}

        save_subdir = output_root / f"{dataset_name}_{split}"
        save_subdir.mkdir(parents=True, exist_ok=True)
        pattern = str(save_subdir / "shard-%06d.tar")

        with wds.ShardWriter(pattern, maxcount=int(args.maxcount), maxsize=float(args.maxsize)) as sink:
            for img_id, anns in tqdm(img2anns.items(), desc=f"Processing {dataset_name}"):
                img_info = img_info_map[img_id]
                rel_path = img_info["file_path"]
                if rel_path not in image_map:
                    raise ValueError(f"Image path {rel_path} not found in image map {image_map_path}.")

                feat_path = feature_root / image_map[rel_path]
                if not feat_path.exists():
                    raise ValueError(f"Feature path {feat_path} does not exist.")
                features = torch.load(feat_path, map_location="cpu")

                orig_w, orig_h = img_info["width"], img_info["height"]
                longest = max(orig_w, orig_h)
                scale = dino_img_size / float(longest)
                new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
                pad_left = (dino_img_size - new_w) // 2
                pad_top = (dino_img_size - new_h) // 2

                padding_mask = torch.ones((dino_img_size, dino_img_size), dtype=torch.bool)
                padding_mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = False

                img_info_save = {
                    "padding_mask": padding_mask,
                    "K": torch.tensor(img_info["K"], dtype=torch.float32),
                    "h": torch.tensor(orig_h, dtype=torch.float32),
                    "scale": scale,
                    "pad_left": pad_left,
                    "pad_top": pad_top,
                    "path": rel_path,
                }

                pad_info = (pad_top, pad_left, new_h, new_w, scale)
                targets_list = [process_sample(ann, pad_info, img_info, dino_img_size) for ann in anns]
                key = f"{dataset_name}_{split}_{rel_path.replace('/', '_').replace('.', '_')}"
                sample = {
                    "__key__": key,
                    "feat.pth": features,
                    "targets.pth": targets_list,
                    "img_info.pth": img_info_save,
                }
                try:
                    sink.write(sample)
                except Exception as exc:
                    print(f"Error writing sample {key}: {exc}")
                else:
                    if args.cleanup_features:
                        feat_path.unlink()

    print("[ConvertToWDS] completed.")


def parse_args():
    parser = ArgumentParser(description="Convert MoCA3D json annotations + features to WebDataset shards")
    parser.add_argument("--config", type=str, default=str(MOCA_ROOT / "configs" / "MoCA_config.yaml"))
    parser.add_argument("--json-file", type=str, default=str(MOCA_ROOT / "datasets" / "MoCA3D"))
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--feature-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--image-map-path", type=str, default=None)
    parser.add_argument("--target-quality", type=str, default=None)
    parser.add_argument("--dino-img-size", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--maxcount", type=int, default=1000)
    parser.add_argument("--maxsize", type=float, default=3e9)
    parser.add_argument("--cleanup-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
