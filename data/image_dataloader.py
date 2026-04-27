from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, WeightedRandomSampler
import cv2
import json
import random
import functools

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from .data_utils import balanced_sampler, filtered_annotations, _LetterBoxing, DistributedWeightedSampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _dataset_name_from_json_path(json_path: Path) -> str:
    return json_path.stem.rsplit("_", 1)[0]

class ImageTransform:
    def __init__(self, image_size):
        self.image_size = image_size
    def __call__(self, image):
        if self.image_size and self.image_size > 0:
            image = _LetterBoxing(image, self.image_size)
        arr = torch.from_numpy(np.array(image, copy=True)).float() / 255.0
        arr = arr.permute(2, 0, 1)  # HWC -> CHW
        mean = arr.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = arr.new_tensor(IMAGENET_STD).view(3, 1, 1)
        return (arr - mean) / std
    
def _default_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    """
    Resize to a square (if requested), convert to CHW float tensor, and normalize.
    """
    return ImageTransform(image_size)

def _seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
class AnnotationDataset(Dataset):
    """
    Minimal dataset that collects every image file under a root directory.
    """

    def __init__(
        self,
        json_data: List[Dict],
        root_dir: str | Path,
        transform_da3: Callable[[Image.Image], torch.Tensor],
        transform_dino: Callable[[Image.Image], torch.Tensor],
        dino_image_size: int = 512,
    ) -> None:
        self.root = Path(root_dir).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        self.transform_da3 = transform_da3
        self.transform_dino = transform_dino
        self.dino_image_size = dino_image_size

        self.ann_list = []
        self.image_map = {}

        for data in json_data:
            dataset_name = data.get("_dataset_name", "unknown")
            for img_info in data["images"]:
                image_key = (dataset_name, img_info["id"])
                self.image_map[image_key] = {"path": img_info["file_path"], "K": img_info["K"]}
            for obj in data["annotations"]:
                self.ann_list.append({
                    "image_key": (dataset_name, obj["image_id"]),
                    "2d_bbx": obj["bbox2D_tight"],
                    "3d_bb8": obj["projected_corners"],
                    "depth": obj["depth"],
                    "3d_bbx": obj["bbox3D_cam"],
                    "index_mapping": obj.get("index_mapping", None),
                })

    def __len__(self) -> int:
        return len(self.ann_list)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        # Processing input images
        ann_data = self.ann_list[index]
        img_info = self.image_map[ann_data["image_key"]]
        img_path = self.root / img_info["path"]
        image = Image.open(img_path).convert("RGB")

        image_da3 = self.transform_da3(image)
        image_dino = self.transform_dino(image)

        # Applying Letterboxing transformations to bounding boxes
        w, h = image.size
        longest = max(w, h)
        scale = self.dino_image_size / float(longest)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        pad_left = (self.dino_image_size - new_w) // 2
        pad_top = (self.dino_image_size - new_h) // 2
        
        raw_corners = torch.tensor([(float(c["u"]), float(c["v"])) for c in ann_data["3d_bb8"]], dtype=torch.float32)
        gt_corners = raw_corners * scale
        gt_corners[:, 0] += pad_left
        gt_corners[:, 1] += pad_top
        gt_corners = gt_corners / self.dino_image_size  
        #### Normalize to [0,1] ####
        
        # 2D Bounding Box
        bbx2d = torch.tensor(ann_data["2d_bbx"], dtype=torch.float32) * scale
        bbx2d[[0,2]] += pad_left
        bbx2d[[1,3]] += pad_top
        bbx2d = bbx2d / self.dino_image_size
        #### Normalize to [0,1] ####
        
        # depths
        raw_depths = torch.clamp(torch.tensor(ann_data["depth"], dtype=torch.float32), min=1e-3)
        
        # Apply Virtual Depth Normalization
        # Target: Virtual Camera with f_v=512, H_v=512
        # Formula: Z_v = Z_real * (f_v / f_real) * (H_real / H_v)
        f_y = img_info["K"][1][1]
        H_real = h 
        f_v = 512.0
        H_v = 512.0
        
        virtual_scale = (f_v / f_y) * (H_real / H_v)
        raw_depths = raw_depths * virtual_scale
        
        padding_mask = torch.ones((self.dino_image_size, self.dino_image_size), dtype=torch.bool)
        padding_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = False
        
        return {
            # Input Values
            "image_da3": image_da3, "image_dino": image_dino, "path": str(img_info["path"]), "2d_bbx": bbx2d,
            # GT infos
            "gt_corners": gt_corners,  # Normalized [0,1]
            "gt_depths": raw_depths,  # Log space depths
            # padding mask
            "padding_mask": padding_mask,
            # for evaluation
            "K": torch.tensor(img_info["K"], dtype=torch.float32),
            'h': H_real,
            "3d_bbx": torch.tensor(ann_data["3d_bbx"], dtype=torch.float32),
            "pad_left": pad_left, "pad_top": pad_top, "scale": scale,
            "index_mapping": torch.tensor(ann_data["index_mapping"] if ann_data["index_mapping"] is not None else list(range(8)), dtype=torch.long),
            }
    


def build_image_dataloader(
    root_dir: Path,
    data_dir: Path,
    seed: int,
    split: str = "test",
    filter: bool = True, 
    batch_size: int = 8,
    da3_image_size: int = 448,
    dino_image_size: int = 512,
    target_quality: str = "Good",
    min_area: int = 32*32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    datasets: list = None,
    epoch_length: int | None = None,
    is_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader from a directory that only contains images.
    """
    if datasets is not None:
        json_paths = []
        missing_datasets = []
        for dataset in datasets:
            matched_paths = sorted(root_dir.glob(f"{dataset}_{split}.json"))
            if not matched_paths:
                missing_datasets.append(dataset)
                continue
            json_paths.extend(matched_paths)
        if missing_datasets:
            raise ValueError(
                f"Requested datasets missing for split '{split}': {missing_datasets}. "
                f"Looked under {root_dir}"
            )
    else:
        json_paths = sorted(root_dir.glob(f"*_{split}.json"))
    if not json_paths:
        raise ValueError(f"No json files found for split '{split}' under {root_dir}")
    sampler = None
    json_list = []
    dataset_annotation_counts = {}
    for json_path in json_paths:
        if filter == True:
            data = filtered_annotations(
                json_path,
                target_quality=target_quality,
                min_area=min_area,
                dino_img_size=dino_image_size,
            )
        else:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        dataset_name = _dataset_name_from_json_path(json_path)
        data["_dataset_name"] = dataset_name
        dataset_annotation_counts[dataset_name] = len(data.get("annotations", []))
        json_list.append(data)
    if rank == 0:
        print(f"[ImageDataloader] split={split} json_paths: {[p.name for p in json_paths]}")
        print(f"[ImageDataloader] split={split} filtered annotation counts: {dataset_annotation_counts}")
    requested_empty = [name for name, count in dataset_annotation_counts.items() if count == 0]
    if requested_empty:
        raise ValueError(
            f"No usable annotations after filtering for split '{split}': {requested_empty}. "
            f"target_quality={target_quality}, min_area={min_area}, dino_image_size={dino_image_size}"
        )
    if split == "train":
        sampler = balanced_sampler(
            json_paths,
            json_list,
            is_ddp=is_ddp,
            rank=rank,
            world_size=world_size,
            epoch_length=epoch_length,
        )
        shuffle = False
    dataset = AnnotationDataset(
        json_data=json_list,
        root_dir=data_dir,
        transform_da3=_default_transform(da3_image_size),
        transform_dino=_default_transform(dino_image_size),
        dino_image_size=dino_image_size,
    )
    if split != "train":
        if epoch_length is not None:
            replacement = epoch_length > len(dataset)
            weights = torch.ones(len(dataset), dtype=torch.double)
            if is_ddp:
                sampler = DistributedWeightedSampler(
                    weights=weights,
                    num_replicas=world_size,
                    rank=rank,
                    replacement=replacement,
                    num_samples=epoch_length,
                )
            else:
                sampler = WeightedRandomSampler(
                    weights,
                    num_samples=epoch_length,
                    replacement=replacement,
                )
            shuffle = False
        elif is_ddp:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            shuffle = False
    
    generator = torch.Generator().manual_seed(seed + rank)
    worker_init = functools.partial(_seed_worker, base_seed=seed)
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=worker_init,
    )


if __name__ == "__main__":
    # Example: iterate over a folder full of images
    sample_root = Path("/home/vmg/Desktop/layout2video/datasets/L2V_new")
    data_root = Path("/home/vmg/Desktop/layout2video/datasets")
    loader = build_image_dataloader(sample_root, data_root, split="val", filter=True, batch_size=2, da3_image_size=448, dino_image_size=512, num_workers=0)
    batch = next(iter(loader))
    print("Batch DA3 image tensor shape:", batch["image_da3"].shape)
    print("Batch DINO image tensor shape:", batch["image_dino"].shape)
    print("Batch file paths:", batch["path"])
    print("min/max DA3 image pixel values:", batch["image_da3"].min().item(), batch["image_da3"].max().item())
    print("min/max DINO image pixel values:", batch["image_dino"].min().item(), batch["image_dino"].max().item())
    print("length of dataset:",len(loader.dataset))
