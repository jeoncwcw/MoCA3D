<div align="center">

<h2 align="center">
  MoCA3D: Monocular 3D Bounding Box Prediction in the Image Plane
</h2>

<p align="center">
  <strong>Changwoo Jeon</strong><sup>1,2</sup> &nbsp;
  <strong>Rishi Upadhyay</strong><sup>1</sup> &nbsp;
  <strong>Achuta Kadambi</strong><sup>1</sup> &nbsp;
</p>

<p align="center">
  <sup>1</sup>UCLA &nbsp;&nbsp;
  <sup>2</sup>Yonsei University
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.19538"><img src="https://img.shields.io/badge/arXiv-2603.19538-b31b1b.svg" alt="arXiv"></a>
  <a href="https://jeoncwcw.github.io/moca3d/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/jeoncwcw/MoCA3D"><img src="https://img.shields.io/badge/🤗-Model_Weights-yellow.svg" alt="HuggingFace"></a>
</p>

</div>

---

MoCA3D is a monocular, class-agnostic 3D object geometry model that predicts projected 3D cuboid corners and per-corner depths from a single RGB image and a tight 2D bounding box, without requiring camera intrinsics at inference time. Rather than lifting an RoI to a compact 3D parameterization, MoCA3D casts geometry recovery as dense prediction with corner heatmaps and depth maps, targeting image-plane geometric fidelity for downstream applications.

To evaluate this setting, the paper introduces Pixel-Aligned Geometry (PAG), a geometry-centric metric suite that measures projected-corner consistency in the image plane together with depth accuracy at the corners.

## 1. Installation

Create the MoCA3D environment from the repository root:

```bash
conda env create -f environment.yml
conda activate moca3d
```

Install the PyTorch stack used for this release:

```bash
pip install --upgrade pip
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
```

Install PyTorch3D and its runtime dependencies:

```bash
pip install fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" --no-build-isolation
```

You can verify the main entrypoints with:

```bash
python tools/evaluate.py --help
python tools/evaluate_cube.py --help
python tools/train.py --help
python tools/train_cube.py --help
```

## 1.1 Optional SAM2 Installation for Missing `bbox2D_tight`

Some public Omni3D sources do not provide `bbox2D_tight` for every annotation. MoCA3D includes an optional SAM2-based preprocessing step that can fill missing `bbox2D_tight` locally from the public assets, using the projected 3D corners as the SAM2 box prompt.

To use this preprocessing path, install the official SAM2 package into the same environment:

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
pip install -e third_party/sam2
```

Download a SAM2 checkpoint such as `sam2.1_hiera_large.pt`, and pass the checkout and checkpoint paths to the preprocessing command with `--sam2-repo-root` and `--sam2-checkpoint`.

## 2. Dataset Preparation

Prepare the Omni3D data following the official Omni3D `DATA.md` instructions.

The repository expects the following directory layout:

```text
datasets/
  KITTI_object/
  SUNRGBD/
  Hypersim/
  ARKitScenes/
  nuScenes/
  Objectron/
  MoCA3D/
    KITTI_train.json
    KITTI_val.json
    KITTI_test.json
    SUNRGBD_train.json
    SUNRGBD_val.json
    SUNRGBD_test.json
    ...
```

Notes:

- Raw dataset assets are expected under `datasets/`.
- MoCA3D annotations are expected under `datasets/MoCA3D/`.
- Annotation files must follow the naming convention `<dataset>_<split>.json`.
- Dataset directories may be regular directories or symlinks, as long as the paths referenced by the annotations resolve correctly.
- The provided configs assume this layout.

## 3. Checkpoint Download

MoCA3D uses a DINOv3 backbone together with the released MoCA3D checkpoint. Download the DINOv3 `ViT-L/16 distilled` checkpoint pretrained on `LVD-1689M` from the official DINOv3 repository:

- https://github.com/facebookresearch/dinov3

Place the DINOv3 checkpoint at:

```text
checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Download the pretrained MoCA3D checkpoint from Hugging Face:

- https://huggingface.co/jeoncwcw/MoCA3D

Place the MoCA3D checkpoint at:

```text
checkpoints/moca3d.safetensors
```

If you use these paths, the default configs can be used without modification.

## 4. MoCA3D Annotation Generation

After preparing the Omni3D data, generate MoCA3D annotations by running:

```bash
bash scripts/omni_to_moca.sh
```

This script generates split-wise MoCA3D annotation files under `datasets/MoCA3D/`.

The default pipeline does:

1. project `bbox3D_cam` into `projected_corners` and extract per-corner `depth`,
2. assign quality groups (`Good`, `Moderate`, `Poor`),
3. reorder projected corners into MoCA3D's canonical 2D ordering.

MoCA3D expects annotations with the following structure:

```text
dataset {
    "info"                  : info,
    "images"                : [image],
    "categories"            : [category],
    "annotations"           : [object],
}

info {
    "id"                    : str,
    "source"                : int,
    "name"                  : str,
    "split"                 : str,
    "version"               : str,
    "url"                   : str,
}

image {
    "id"                    : int,
    "dataset_id"            : int,
    "width"                 : int,
    "height"                : int,
    "file_path"             : str,
    "K"                     : list (3x3),
    "src_90_rotate"         : int,                    # image rotated X times, 90 deg counterclockwise
    "src_flagged"           : bool,                   # flagged as potentially inconsistent sky direction
}

category {
    "id"                    : int,
    "name"                  : str,
    "supercategory"         : str
}

object {
    "id"                    : int,                    # unique annotation identifier
    "image_id"              : int,                    # identifier for image
    "category_id"           : int,                    # identifier for the category
    "category_name"         : str,                    # plain name for the category

    # General 2D/3D Box Parameters.
    # Values are set to -1 when unavailable.
    "valid3D"               : bool,                   # flag for no reliable 3D box
    "bbox2D_tight"          : [x1, y1, x2, y2],      # 2D corners of annotated tight box
    "bbox2D_proj"           : [x1, y1, x2, y2],      # 2D corners projected from bbox3D
    "bbox2D_trunc"          : [x1, y1, x2, y2],      # 2D corners projected from bbox3D then truncated
    "bbox3D_cam"            : [[x1, y1, z1]...[x8, y8, z8]]
    "center_cam"            : [x, y, z],             # 3D center in meters and camera coordinates
    "dimensions"            : [width, height, length],   # object dimensions in meters
    "R_cam"                 : list (3x3),           # 3D rotation matrix to the camera frame rotation

    # Optional dataset specific properties,
    # used mainly for evaluation and ignore.
    # Values are set to -1 when unavailable.
    "behind_camera"         : bool,                  # a corner is behind camera
    "visibility"            : float,                 # annotated visibility 0 to 1
    "truncation"            : float,                 # computed truncation 0 to 1
    "segmentation_pts"      : int,                   # visible instance segmentation points
    "lidar_pts"             : int,                   # visible LiDAR points in the object
    "depth_error"           : float,                 # L1 of depth map and rendered object

    # MoCA3D
    "projected_corners"     : [["u": int, "v": int], ...],
    "depth"                 : [d1, d2, ...],
    "quality"               : str("Good", "Moderate", or "Poor"),
}
```

### 4.1 Optional: Fill Missing `bbox2D_tight` with SAM2

Public Omni3D annotations for datasets such as `ARKitScenes`, `nuScenes`, and `Objectron` may not include `bbox2D_tight`. MoCA3D provides an optional SAM2-based preprocessing script, `data/preprocess/fill_missing_bbox2d_sam2.py`, that fills missing `bbox2D_tight` entries locally after the public images and MoCA3D-format json files are prepared.

The script uses each object's projected 3D corners to form a coarse 2D prompt box, runs SAM2 on the corresponding image, and writes the tight mask bounding box back to `bbox2D_tight`. By default, it only updates annotations whose `bbox2D_tight` is missing or invalid.

Run it directly with:

```bash
python data/preprocess/fill_missing_bbox2d_sam2.py \
  --input-dir datasets/MoCA3D \
  --output-dir datasets/MoCA3D \
  --data-root datasets \
  --datasets ARKitScenes nuScenes Objectron \
  --splits train val test \
  --sam2-repo-root third_party/sam2 \
  --sam2-checkpoint /path/to/sam2.1_hiera_large.pt
```

If you want the full conversion script to call this SAM2 step after generating MoCA3D annotations, enable it through environment variables:

```bash
USE_SAM2_BBOX_FILL=1 \
SAM2_REPO_ROOT=third_party/sam2 \
SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt \
bash scripts/omni_to_moca.sh
```

The defaults target `ARKitScenes`, `nuScenes`, and `Objectron` over the `train`, `val`, and `test` splits. Override them when needed:

```bash
USE_SAM2_BBOX_FILL=1 \
SAM2_REPO_ROOT=third_party/sam2 \
SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt \
SAM2_DATASETS="ARKitScenes nuScenes" \
SAM2_SPLITS="train val" \
bash scripts/omni_to_moca.sh
```

Use `--process-all` with `fill_missing_bbox2d_sam2.py` only if you intentionally want to recompute every `bbox2D_tight` entry in the selected files.

The SAM2 step is a preprocessing helper, not a ground-truth replacement. If you use it for training or evaluation on a new dataset release, inspect the filled boxes and keep the exact script/checkpoint combination fixed for reproducibility.

## 5. Optional WDS Data Preparation

To build WDS-format data for a specific split, run:

```bash
bash scripts/build_moca_wds.sh <split>
```

Supported values for `<split>` are:

- `train`
- `val`
- `test`

Generated shards are written under `datasets/moca_wds/`.

## 6. Run Training and Evaluation

The repository provides the following entrypoint scripts:

- `bash scripts/moca_train.sh` for MoCA3D training
- `bash scripts/moca_evaluate.sh` for MoCA3D evaluation
- `bash scripts/moca_cube_train.sh` for MoCA3D-Cube training
- `bash scripts/moca_cube_evaluate.sh` for MoCA3D-Cube evaluation

With the checkpoints from Section 3 and the evaluation scripts, users can reproduce the reported evaluation results for the datasets whose public assets and annotations are prepared in the expected format. Training scripts are included for completeness and can be used once the required public dataset assets and annotation files are prepared.
