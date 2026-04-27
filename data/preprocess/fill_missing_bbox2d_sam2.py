import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_DATASETS = ["ARKitScenes", "nuScenes", "Objectron"]
DEFAULT_SPLITS = ["train", "val", "test"]


def _normalize_name_list(values):
    normalized = []
    for value in values or []:
        for token in str(value).split(","):
            token = token.strip().strip("[]").strip().strip("'\"")
            if token:
                normalized.append(token)
    seen = set()
    deduped = []
    for value in normalized:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _load_sam2_predictor(sam2_repo_root: Path | None, config_file: str, checkpoint_path: Path, device: str):
    if sam2_repo_root is not None:
        sys.path.insert(0, str(sam2_repo_root))

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            "Unable to import SAM2. Install the official SAM2 package or pass --sam2-repo-root "
            "pointing to a local SAM2 checkout."
        ) from exc

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

    sam2_model = build_sam2(
        config_file=config_file,
        ckpt_path=str(checkpoint_path),
        device=device,
        apply_postprocessing=False,
    )
    return SAM2ImagePredictor(sam2_model)


def _iter_json_paths(input_dir: Path, datasets, splits):
    paths = []
    for dataset_name in datasets:
        for split in splits:
            json_path = input_dir / f"{dataset_name}_{split}.json"
            if json_path.exists():
                paths.append(json_path)
    return paths


def _has_missing_bbox(obj):
    bbox = obj.get("bbox2D_tight")
    if bbox is None or len(bbox) != 4:
        return True
    return any(float(value) < 0 for value in bbox)


def _rough_box_from_projected_corners(corners, width: int, height: int):
    if len(corners) != 8:
        return None
    try:
        u_coords = [float(corner["u"]) for corner in corners]
        v_coords = [float(corner["v"]) for corner in corners]
    except (KeyError, TypeError, ValueError):
        return None

    x1, x2 = min(u_coords), max(u_coords)
    y1, y2 = min(v_coords), max(v_coords)
    x1 = max(0.0, min(x1, width - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _mask_to_bbox(mask):
    y_idx, x_idx = np.where(mask)
    if y_idx.size == 0 or x_idx.size == 0:
        return None
    x1, x2 = float(np.min(x_idx)), float(np.max(x_idx))
    y1, y2 = float(np.min(y_idx)), float(np.max(y_idx))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [x1, y1, x2, y2]


def _process_json_file(predictor, json_path: Path, output_dir: Path, data_root: Path, only_missing: bool):
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    image_map = {image["id"]: image for image in data.get("images", [])}
    object_map = defaultdict(list)
    for obj in data.get("annotations", []):
        if only_missing and not _has_missing_bbox(obj):
            continue
        object_map[obj["image_id"]].append(obj)

    stats = {
        "images_scanned": 0,
        "objects_filled": 0,
        "objects_failed": 0,
        "objects_skipped": 0,
        "objects_missing_before": sum(len(items) for items in object_map.values()),
    }
    if not object_map:
        output_path = output_dir / json_path.name
        if output_path != json_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
        return stats

    for image_id, objects in tqdm(
        object_map.items(),
        total=len(object_map),
        desc=f"[SAM2] {json_path.name}",
        leave=False,
    ):
        image_info = image_map.get(image_id)
        if image_info is None:
            stats["objects_skipped"] += len(objects)
            continue

        image_path = data_root / image_info["file_path"]
        if not image_path.exists():
            stats["objects_skipped"] += len(objects)
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            stats["objects_skipped"] += len(objects)
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        stats["images_scanned"] += 1

        img_width = int(image_info.get("width", image_rgb.shape[1]))
        img_height = int(image_info.get("height", image_rgb.shape[0]))

        for obj in objects:
            rough_box = _rough_box_from_projected_corners(obj.get("projected_corners", []), img_width, img_height)
            if rough_box is None:
                stats["objects_failed"] += 1
                continue

            try:
                with torch.inference_mode():
                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=rough_box[None, :],
                        multimask_output=False,
                    )
                bbox = _mask_to_bbox(masks[0])
            except Exception:
                bbox = None

            if bbox is None:
                stats["objects_failed"] += 1
                continue

            obj["bbox2D_tight"] = bbox
            stats["objects_filled"] += 1

    output_path = output_dir / json_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill missing bbox2D_tight entries with SAM2 using projected corners as the prompt box."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing <dataset>_<split>.json files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for updated json files. Defaults to --input-dir for in-place updates.",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root used to resolve image file_path entries.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, help="Datasets to process.")
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS, help="Splits to process.")
    parser.add_argument(
        "--sam2-repo-root",
        type=Path,
        default=None,
        help="Optional path to a local SAM2 checkout. Not needed if SAM2 is already installed.",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default=DEFAULT_SAM2_CONFIG,
        help="SAM2 config passed to build_sam2.",
    )
    parser.add_argument("--sam2-checkpoint", type=Path, required=True, help="Path to the SAM2 checkpoint.")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--process-all", action="store_true", help="Recompute bbox2D_tight for every annotation, not just missing ones.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.datasets = _normalize_name_list(args.datasets)
    args.splits = _normalize_name_list(args.splits)
    output_dir = args.output_dir or args.input_dir

    predictor = _load_sam2_predictor(
        sam2_repo_root=args.sam2_repo_root,
        config_file=args.sam2_config,
        checkpoint_path=args.sam2_checkpoint,
        device=args.device,
    )

    json_paths = _iter_json_paths(args.input_dir, args.datasets, args.splits)
    if not json_paths:
        raise ValueError(
            f"No matching json files found under {args.input_dir} for datasets={args.datasets} and splits={args.splits}"
        )

    total = defaultdict(int)
    for json_path in json_paths:
        stats = _process_json_file(
            predictor=predictor,
            json_path=json_path,
            output_dir=output_dir,
            data_root=args.data_root,
            only_missing=not args.process_all,
        )
        for key, value in stats.items():
            total[key] += value
        print(
            f"[SAM2] {json_path.name}: "
            f"missing_before={stats['objects_missing_before']} "
            f"filled={stats['objects_filled']} "
            f"failed={stats['objects_failed']} "
            f"skipped={stats['objects_skipped']}"
        )

    print(
        "[SAM2] Summary: "
        f"images_scanned={total['images_scanned']} "
        f"missing_before={total['objects_missing_before']} "
        f"filled={total['objects_filled']} "
        f"failed={total['objects_failed']} "
        f"skipped={total['objects_skipped']}"
    )


if __name__ == "__main__":
    main()
