import torch
import webdataset as wds
import math
from pathlib import Path
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
import random
import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)
def _worker_init_fn(worker_id: int):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

DATASET_STATS_ANN_TRAIN = {
    "ARKitScenes": {"count": 30850,   "type": "indoor"},
    "Hypersim":    {"count": 250344,  "type": "indoor"},
    "KITTI":       {"count": 4435,    "type": "urban"},
    "nuScenes":    {"count": 37297,   "type": "urban"},
    "Objectron":   {"count": 28887,   "type": "general"},
    "SUNRGBD":     {"count": 12130,   "type": "indoor"},
}
DATASET_STATS_IMAGE_TRAIN = {
    "ARKitScenes": {"count": 21505,   "type": "indoor"},
    "Hypersim":    {"count": 42996,   "type": "indoor"},
    "KITTI":       {"count": 2045,    "type": "urban"},
    "nuScenes":    {"count": 17118,   "type": "urban"},
    "Objectron":   {"count": 25703,   "type": "general"},
    "SUNRGBD":     {"count": 3828,    "type": "indoor"},
}

class FeatureGeometryAug:
    def __init__(self, cfg):
        self.flip_prob = cfg.aug.flip_prob
        self.rot_range = cfg.aug.rot_range
        self.swap_map = [3, 2, 1, 0, 7, 6, 5, 4]
        self.cfg_noise = cfg.aug.feature_noise_sigma
        self.cfg_jitter = cfg.aug.box_jitter_sigma
        
        self.current_noise = 0.0
        self.current_jitter = 0.0
    
    def set_epoch(self, epoch):
        if epoch < 5:
            self.current_noise = 0.0
            self.current_jitter = 0.0
        else:
            self.current_noise = self.cfg_noise
            self.current_jitter = self.cfg_jitter
        
    def __call__(self, samples):
        for sample in samples:
            if self.rot_range > 0:
                angle = (torch.rand(1).item() * 2 - 1) * self.rot_range
                rad = math.radians(angle)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                
                # Feature rotation
                for key in ["feat_dino"]:
                    sample[key] = TF.rotate(sample[key], angle, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
                # Corner rotation
                curr_corners = sample["gt_corners"] - 0.5
                new_corners = torch.empty_like(curr_corners)
                new_corners[:, 0] = curr_corners[:, 0] * cos_a - curr_corners[:, 1] * sin_a
                new_corners[:, 1] = curr_corners[:, 0] * sin_a + curr_corners[:, 1] * cos_a
                sample["gt_corners"] = (new_corners + 0.5).clamp(0, 1)
                
                # Box rotation
                x1, y1, x2, y2 = sample["2d_bbx"]
                bx = torch.tensor([x1, x2, x2, x1]) - 0.5
                by = torch.tensor([y1, y1, y2, y2]) - 0.5
                new_bx = bx * cos_a - by * sin_a + 0.5
                new_by = bx * sin_a + by * cos_a + 0.5
                sample["2d_bbx"] = torch.tensor([new_bx.min(), new_by.min(), new_bx.max(), new_by.max()]).clamp(0, 1)
                # padding rotation
                mask_float = sample["padding_mask"].float().unsqueeze(0) # [1, H, W]
                rotated_mask = TF.rotate(mask_float, angle, interpolation=TF.InterpolationMode.NEAREST, expand=False, fill=1.0)
                sample["padding_mask"] = rotated_mask.squeeze(0) > 0.5
                
            if torch.rand(1) < self.flip_prob:
                for key in ["feat_dino", "padding_mask"]:
                    sample[key] = TF.hflip(sample[key])
                
                sample["gt_corners"][:, 0] = 1.0 - sample["gt_corners"][:, 0].clamp(0, 1)
                sample["gt_corners"] = sample["gt_corners"][self.swap_map]
                sample["gt_depths"] = sample["gt_depths"][self.swap_map]  # Also swap depths
                x1, y1, x2, y2 = sample["2d_bbx"]
                sample["2d_bbx"] = torch.tensor([1.0 - x2, y1, 1.0 - x1, y2])
            if self.current_noise > 0:
                for key in ["feat_dino"]:
                    sample[key] += torch.randn_like(sample[key]) * self.current_noise
            if self.current_jitter > 0:
                noise = torch.randn_like(sample["2d_bbx"]) * self.current_jitter
                sample["2d_bbx"] = torch.clamp(sample["2d_bbx"] + noise, 0, 1)
            yield sample
        
class FlattenSamples:
    def __init__(self, dino_img_size: int = 512, max_cap: int = 4):
        self.dino_img_size = dino_img_size
        self.max_cap = max_cap
    def __call__(self, samples):
        for sample in samples:
            if not sample["targets.pth"]:
                continue
            features = sample["feat.pth"]
            img_info = sample["img_info.pth"]
        
            f_dino = features["dino"].float().squeeze(0)
            
            # If max_cap is negative or large, use all targets (no subsampling)
            if self.max_cap > 0 and len(sample["targets.pth"]) > self.max_cap:
                selected_targets = random.sample(sample["targets.pth"], self.max_cap)
            else:
                selected_targets = sample["targets.pth"]
                
            for target in selected_targets:
                gt_corners = torch.as_tensor(target["gt_corners"], dtype=torch.float32)
                gt_depths = torch.as_tensor(target["gt_depths"], dtype=torch.float32)
                bbx2d = torch.as_tensor(target["2d_bbx"], dtype=torch.float32)
                bbx3d = torch.as_tensor(target["3d_bbx"], dtype=torch.float32)
                index_mapping = target.get("index_mapping")
                if index_mapping is None:
                    index_mapping = list(range(8))
                yield {
                    # Feature
                    "feat_dino": f_dino,
                    # GT
                    "gt_corners": gt_corners, "gt_depths": gt_depths,
                    # Bounding Box
                    "2d_bbx": bbx2d, "3d_bbx": bbx3d,
                    # mask for transformer
                    "padding_mask": img_info["padding_mask"],
                    # image parameters
                    "K": torch.as_tensor(img_info["K"], dtype=torch.float32), "h": img_info["h"], "scale": img_info["scale"],
                    "pad_left": img_info["pad_left"], "pad_top": img_info["pad_top"],
                    # for evaluation
                    "path": img_info["path"],
                    "index_mapping": torch.as_tensor(index_mapping, dtype=torch.long),
                }     
            
def get_balanced_weights(found_datasets, max_cap=4):
    image_stats = DATASET_STATS_IMAGE_TRAIN
    raw_scores = {}
    for name in found_datasets:
        key = next((k for k in image_stats if k in name), None)
        if key:
            img_count = image_stats[key]["count"]
            score = math.sqrt(img_count)
            raw_scores[name] = score
    total_score = sum(raw_scores.values())
    final_weights = {k: v / total_score for k, v in raw_scores.items()}
    
    return final_weights


def _make_wds_url_from_shards(shards):
    # Avoid brace-expansion patterns like shard-{000000..000123}.tar because
    # some environments don't expand them and treat them as literal file names.
    if len(shards) == 1:
        return str(shards[0])
    return [str(s) for s in shards]

    
def build_wds_feature_dataloader(
    cfg: any,
    wds_root: Path,
    split: str = "train",
    enable_aug: bool = True,
    dino_img_size: int = 512,
    epoch_length: int = 50000,
    world_size: int = 1,
    include_datasets: list = None,  # Filter specific datasets
    full_dataset_eval: bool = False,  # Evaluate all test samples without epoch truncation
    random_mix: bool = False,        # Whether to use RandomMix (weighted) or ChainDataset
    max_cap: int = -1,               # Max targets per image (set to -1 to disable subsampling)
):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    wds_root = Path(wds_root)
    dataset_dirs = sorted(list(wds_root.glob(f"*_{split}")))
    
    # Filter by include_datasets
    if include_datasets is not None:
        include_set = {str(name).strip() for name in include_datasets if str(name).strip()}
        dataset_dirs = [
            d for d in dataset_dirs
            if d.name in include_set or d.name.rsplit("_", 1)[0] in include_set
        ]
        print(f"WDS Dataloader - Filtering to datasets: {sorted(include_set)}")
    
    if not dataset_dirs:
        raise ValueError(f"No WDS dataset found in {wds_root} for split {split} (filter: {include_datasets})")
    found_dataset_names = [d.name for d in dataset_dirs]
    
    is_train = (split == "train")
    is_val = (split == "val")
    
    urls = []
    weights = []
    
    # Use random mix only if training AND requested
    if is_train and random_mix:
        # Training: use balanced weights for dataset mixing
        weight_map = get_balanced_weights(found_dataset_names)
        print(f"WDS Dataloader - Using RandomMix with weights: {weight_map}")
        for d_dir in dataset_dirs:
            d_name = d_dir.name
            if d_name not in weight_map:
                continue
            shards = sorted(list(d_dir.glob("shard-*.tar")))
            if not shards: raise ValueError(f"No shards found in {d_dir}")
            url = _make_wds_url_from_shards(shards)
            urls.append(url)
            weights.append(weight_map[d_name])
        sum_w = sum(weights)
        weights = [w / sum_w for w in weights]
    else:
        # Test/Val OR Training without RandomMix: just collect all URLs
        print(f"WDS Dataloader - {split} mode (random_mix={random_mix}), loading datasets: {found_dataset_names}")
        for d_dir in dataset_dirs:
            shards = sorted(list(d_dir.glob("shard-*.tar")))
            if not shards: raise ValueError(f"No shards found in {d_dir}")
            url = _make_wds_url_from_shards(shards)
            urls.append(url)
    
    # Use max_cap from arguments
    actual_cap = max_cap if is_train else -1
    transform = FlattenSamples(dino_img_size=dino_img_size, max_cap=actual_cap)
    
    aug_transform = None
    if is_train and enable_aug:
        aug_transform = FeatureGeometryAug(cfg)

    # For test full-eval, avoid RandomMix by building one concatenated WebDataset.
    if (not is_train) and full_dataset_eval and len(urls) > 1:
        flat_urls = []
        for url in urls:
            if isinstance(url, list):
                flat_urls.extend(url)
            else:
                flat_urls.append(url)
        print(f"WDS Dataloader - Full eval test mode with concatenated datasets: {len(flat_urls)} shards")
        dataset = (
            wds.WebDataset(
                flat_urls,
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
                shardshuffle=False,
                empty_check=False,
            )
            .decode("torch")
            .compose(transform)
        )
    else:
        datasets = []
        for url in urls:
            if is_train:
                # Training: repeat, shuffle, apply augmentations
                ds = (wds.WebDataset(url, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker, shardshuffle=2000, empty_check=False)
                    .repeat()
                    .shuffle(1500)
                    .decode("torch")
                    .compose(transform)
                    .shuffle(100)
                )
                if aug_transform is not None:
                    ds = ds.compose(aug_transform)
            elif is_val:
                # Val: shuffle but no repeat, sample subset
                ds = (wds.WebDataset(url, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker, shardshuffle=2000, empty_check=False)
                    .shuffle(1500)
                    .decode("torch")
                    .compose(transform)
                    .shuffle(200)
                )
            else:
                if full_dataset_eval:
                    # Full test: iterate every sample once (per-rank split handled by webdataset nodesplitter)
                    ds = (wds.WebDataset(url, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker, shardshuffle=False, empty_check=False)
                        .decode("torch")
                        .compose(transform)
                    )
                else:
                    # Legacy behavior: repeat+epoch trimming
                    # (trimmed later in evaluation.py)
                    ds = (wds.WebDataset(url, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker, shardshuffle=False, empty_check=False)
                        .repeat() 
                        .decode("torch")
                        .compose(transform)
                    )
            datasets.append(ds)

        if len(datasets) > 1:
            if is_train and random_mix:
                dataset = wds.RandomMix(datasets, weights)
            else:
                # For test/val OR training w/o mix: use RandomMix with uniform weights
                # (ChainDataset doesn't exist in webdataset 1.0.2)
                uniform_weights = [1.0 / len(datasets)] * len(datasets)
                dataset = wds.RandomMix(datasets, uniform_weights)
        else:
            dataset = datasets[0]
    if is_train:
        batches_per_rank = epoch_length // (batch_size * world_size)
        loader = (
            wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                          persistent_workers=True, pin_memory=True, worker_init_fn=_worker_init_fn)
            .batched(batch_size, partial=False)
            .with_epoch(batches_per_rank)
        )
    elif is_val:
        # Val: shuffle and limit to val_epoch_length
        val_batches = epoch_length // (batch_size * world_size)
        loader = (
            wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                          persistent_workers=False, pin_memory=True, worker_init_fn=_worker_init_fn)
            .batched(batch_size, partial=True)
            .with_epoch(val_batches)
        )
    else:
        # Test: full-eval mode iterates through all available samples
        if full_dataset_eval:
            loader = (
                wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                              persistent_workers=False, pin_memory=True, worker_init_fn=_worker_init_fn)
                .batched(batch_size, partial=True)
            )
        else:
            # Legacy behavior: trim to a fixed number of batches
            test_batches = epoch_length // (batch_size * world_size)
            loader = (
                wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                              persistent_workers=False, pin_memory=True, worker_init_fn=_worker_init_fn)
                .batched(batch_size, partial=True)
                .with_epoch(test_batches)
            )
    
    return loader, aug_transform

def count_wds_samples(wds_root: Path, split: str = "train") -> dict:
    # Calculate number of samples and annotations in WDS dataset
    import tarfile
    
    wds_root = Path(wds_root)
    dataset_dirs = sorted(list(wds_root.glob(f"*_{split}")))
    
    counts = {}
    total_annotations = 0
    
    for d_dir in dataset_dirs:
        d_name = d_dir.name
        shards = sorted(list(d_dir.glob("shard-*.tar")))
        
        sample_count = 0
        annotation_count = 0
        
        for shard_path in shards:
            with tarfile.open(shard_path, 'r') as tar:
                # Each sample is distinguished by __key__
                keys = set()
                for member in tar.getmembers():
                    key = member.name.rsplit('.', 1)[0]
                    keys.add(key)
                sample_count += len(keys)
                
                # Calculate number of annotations from targets.pth files
                for member in tar.getmembers():
                    if member.name.endswith('targets.pth'):
                        f = tar.extractfile(member)
                        targets = torch.load(f, map_location='cpu')
                        annotation_count += len(targets)
        
        counts[d_name] = {
            "samples": sample_count,
            "annotations": annotation_count
        }
        total_annotations += annotation_count
    
    counts["_total"] = {"annotations": total_annotations}
    return counts

if __name__ == "__main__":
    
    print("\n" + "=" * 50)
    print("📊 Testing build_wds_feature_dataloader()")
    print("=" * 50)
    
    wds_root = Path("/home/vmg/Desktop/layout2video/datasets/betr_wds")
    cfg = OmegaConf.load("/home/vmg/Desktop/layout2video/src_betr/configs/betr_config.yaml")
    if not wds_root.exists():
        print(f"⚠️  WDS root not found: {wds_root}")
    else:
        print("\n📊 Counting actual samples...")
        # counts = count_wds_samples(wds_root, split="train")
        # for name, info in counts.items():
        #     print(f"  {name}: {info}")
        try:
            # Test filtering and no-mix option
            loader, _ = build_wds_feature_dataloader(
                cfg = cfg,
                wds_root=wds_root,
                split="train",
                batch_size=4,
                num_workers=2,
                epoch_length=10,
                include_datasets=["Objectron"], # EXAMPLE: Only Objectron
                random_mix=False,               # EXAMPLE: No Mixing (Sequential)
                max_cap=-1,                     # EXAMPLE: No target subsampling
            )
            
            print("\n🔄 Checking batches (Objectron only)...")
            for i, batch in enumerate(loader):
                print(f"\nBatch {i}:")
                for k, v in batch.items():
                    shape = v.shape if hasattr(v, 'shape') else len(v)
                    print(f"  {k}: {shape}")
                if i >= 1:
                    break
            print("\n✅ Done!")
        except Exception as e:
            print(f"❌ Error: {e}")
