from pathlib import Path
import numpy as np
from typing import Dict, List, Union
import math
import torch
from torch.utils.data import WeightedRandomSampler, Sampler
import torch.distributed as dist
import json
from PIL import Image
import cv2

DATASET_STATS = {
    "Hypersim": {"type": "indoor", "count": 264154},
    "SUNRGBD": {"type": "indoor", "count": 13006},
    "Objectron": {"type": "indoor", "count": 28890},
    "KITTI": {"type": "outdoor", "count": 4435},
    "nuScenes": {"type": "outdoor", "count": 47893},
}

DATASET_STATS_IMAGE_TRAIN = {
    "ARKitScenes": {"count": 21505, "type": "indoor"},
    "Hypersim": {"count": 42996, "type": "indoor"},
    "KITTI": {"count": 2045, "type": "urban"},
    "nuScenes": {"count": 17118, "type": "urban"},
    "Objectron": {"count": 25703, "type": "general"},
    "SUNRGBD": {"count": 3828, "type": "indoor"},
}

def get_hierarchical_weights(found_datasets: List[str], indoor_prob: float = 0.6, outdoor_prob: float = 0.4) -> dict:
    """
    Hierarchical sampling weights:
    1. Split datasets into indoor/outdoor groups
    2. Assign group-level probability (e.g., 60% indoor, 40% outdoor)
    3. Within each group, use sqrt inverse frequency
    """
    groups = {"indoor": [], "outdoor": []}
    
    for name in found_datasets:
        key = next((k for k in DATASET_STATS if k in name), None)
        if key:
            groups[DATASET_STATS[key]["type"]].append((name, DATASET_STATS[key]["count"]))
    
    final_weights = {}
    group_probs = {"indoor": indoor_prob, "outdoor": outdoor_prob}
    
    for g_name, datasets in groups.items():
        datasets = [(name, count) for name, count in datasets if count > 0]
        if not datasets:
            continue
        
        raw_weights = [math.sqrt(count) for _, count in datasets]
        total_score = sum(raw_weights)
        
        target_prob = group_probs[g_name]
        for (d_name, _), w in zip(datasets, raw_weights):
            final_weights[d_name] = (w / total_score) * target_prob
    
    return final_weights

def get_wds_style_weights(found_datasets: List[str], fallback_counts: Dict[str, int] | None = None) -> dict:
    raw_scores = {}
    missing_datasets = []

    for name in found_datasets:
        key = next((k for k in DATASET_STATS_IMAGE_TRAIN if k in name), None)
        if key:
            raw_scores[name] = math.sqrt(DATASET_STATS_IMAGE_TRAIN[key]["count"])
        else:
            missing_datasets.append(name)

    if fallback_counts is not None:
        for name in missing_datasets:
            count = fallback_counts.get(name, 0)
            if count > 0:
                raw_scores[name] = math.sqrt(count)

    if not raw_scores:
        raise ValueError("No datasets available for WDS-style sampling.")

    total_score = sum(raw_scores.values())
    return {name: score / total_score for name, score in raw_scores.items()}

def balanced_sampler(json_paths: List[Path], json_data_list: List[dict],
                     is_ddp: bool=False, rank: int=0, world_size: int=1,
                     epoch_length: int | None = None,
                     ) -> Union[WeightedRandomSampler, Sampler]:
    """
    WDS-style dataset balancing with annotation-level random sampling.
    """
    sample_dataset_indicies = []
    dataset_counts = {}
    for path, data in zip(json_paths, json_data_list):
        dataset_name = path.stem.split('_')[0]
        num_samples = len(data["annotations"])
        sample_dataset_indicies.extend([dataset_name] * num_samples)
        if num_samples > 0:
            dataset_counts[dataset_name] = num_samples
    
    print(f"[Sampler] Dataset Counts: {dataset_counts}")

    found_datasets = list(dataset_counts.keys())
    dataset_weights = get_wds_style_weights(found_datasets, fallback_counts=dataset_counts)
    print(f"[Sampler] WDS-style Weights: {dataset_weights}")
    
    weights = []
    for dataset_name in sample_dataset_indicies:
        dataset_weight = dataset_weights.get(dataset_name, 0.0)
        sample_count = dataset_counts.get(dataset_name, 1)
        weights.append(dataset_weight / sample_count)

    weights = torch.tensor(weights, dtype=torch.double)
    num_samples = epoch_length if epoch_length is not None else len(weights)
    if num_samples <= 0:
        raise ValueError(f"epoch_length must be positive, got {num_samples}")

    if is_ddp:
        return DistributedWeightedSampler(
            weights=weights,
            num_replicas=world_size,
            rank=rank,
            replacement=True,
            num_samples=num_samples,
        )
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

def filtered_annotations(json_path: Path, target_quality: str = "Good", dino_img_size=512, min_area=1024) -> dict:
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = data.get("annotations", [])
    image_map = {img["id"]: img for img in data.get("images", [])}
    
    filtered_annotations = []
    ignore_names = ['dontcare', 'ignore', 'void']
    
    for obj in annotations:
        # Skip crowd annotations (Omni3D datasets may set this)
        if obj.get("iscrowd", 0):
            continue
        
        # Skip explicitly ignored annotations (set by Omni3D dataset registration)
        if obj.get("ignore", False):
            continue
        
        if obj.get("category_name") in ignore_names:
            continue
        if obj.get("quality") != target_quality:
            continue
        bbox = obj.get("bbox2D_tight")
        if not bbox or -1 in bbox:
            continue
        if obj.get("behind_camera"): 
            continue
        if not bool(obj.get("valid3D", True)):
            continue
        dims = obj.get("dimensions", [1, 1, 1])
        if dims[0] <= 0 or dims[1] <= 0 or dims[2] <= 0:
            continue
        if obj.get("center_cam", [0, 0, 0])[2] > 1e8:
            continue
        if obj.get("lidar_pts", 1) == 0:
            continue
        if obj.get("segmentation_pts", 1) == 0:
            continue
        # Default thresholds from Omni3D (Base.yaml)
        # TRUNCATION_THRES = 0.75
        # VISIBILITY_THRES = 0.25
        # DEPTH_ERROR_THRES = 0.5
        if obj.get("depth_error", 0.0) > 0.5:
            continue
        if obj.get("truncation", 0) >= 0.75:
            continue           
        vis = obj.get("visibility", 1.0)
        if vis >= 0 and vis <= 0.25:
             continue
            
        img = image_map[obj["image_id"]]
        img_width, img_height = img["width"], img["height"]
        
        box_h = bbox[3] - bbox[1]
        if box_h <= (0.05 * img_height):
             continue
        if box_h >= (1.5 * img_height):
             continue
        img = image_map[obj["image_id"]]
        img_width, img_height = img["width"], img["height"]
        longest = max(img_width, img_height)
        scale = dino_img_size / float(longest)
        bbox = np.array(bbox, dtype=np.float32)
        bbox = bbox * scale
        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]
        box_area = box_width * box_height
        if box_area >= min_area:
            filtered_annotations.append(obj)
    
    data["annotations"] = filtered_annotations
    return data

class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, num_replicas=None, rank=None, replacement=True, seed=0, num_samples=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        
        total_len = len(self.weights)
        requested_num_samples = total_len if num_samples is None else int(num_samples)
        if requested_num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {requested_num_samples}")
        self.num_samples = math.ceil(requested_num_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indicies = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        indicies = indicies[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        if not self.replacement and len(indicies) < self.num_samples:
             indicies += indicies[:(self.num_samples - len(indicies))]
        return iter(indicies)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch

def _LetterBoxing(img: Image.Image, target_size: int) -> Image.Image:
    target_size = int(target_size)
    if target_size <= 0:
        raise ValueError("target_size must be a positive integer")
    w, h = img.size
    longest = max(w, h)
    if longest == target_size:
        return img
    scale = target_size / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
    delta_w = int(target_size - new_w)
    delta_h = int(target_size - new_h)
        
    pad_left = int(delta_w // 2); pad_right = int(delta_w - pad_left)
    pad_top = int(delta_h // 2); pad_bottom = int(delta_h - pad_top)
    padded_arr = cv2.copyMakeBorder(
        arr,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0,0,0)
    )
    return Image.fromarray(padded_arr)
