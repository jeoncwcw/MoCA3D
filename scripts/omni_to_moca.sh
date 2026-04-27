#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${1:-${ROOT_DIR}/datasets/Omni3D}"
OUTPUT_DIR="${2:-${ROOT_DIR}/datasets/MoCA3D}"

python3 "${ROOT_DIR}/data/preprocess/dataprocess.py" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}"

if [[ "${USE_SAM2_BBOX_FILL:-0}" == "1" ]]; then
  if [[ -z "${SAM2_CHECKPOINT:-}" ]]; then
    echo "USE_SAM2_BBOX_FILL=1 requires SAM2_CHECKPOINT to be set." >&2
    exit 1
  fi

  SAM2_DATASETS=(${SAM2_DATASETS:-ARKitScenes nuScenes Objectron})
  SAM2_SPLITS=(${SAM2_SPLITS:-train val test})

  SAM2_CMD=(
    python3 "${ROOT_DIR}/data/preprocess/fill_missing_bbox2d_sam2.py"
    --input-dir "${OUTPUT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --data-root "${ROOT_DIR}/datasets"
    --datasets "${SAM2_DATASETS[@]}"
    --splits "${SAM2_SPLITS[@]}"
    --sam2-checkpoint "${SAM2_CHECKPOINT}"
  )

  if [[ -n "${SAM2_REPO_ROOT:-}" ]]; then
    SAM2_CMD+=(--sam2-repo-root "${SAM2_REPO_ROOT}")
  fi
  if [[ -n "${SAM2_CONFIG:-}" ]]; then
    SAM2_CMD+=(--sam2-config "${SAM2_CONFIG}")
  fi

  "${SAM2_CMD[@]}"
fi

python3 "${ROOT_DIR}/data/preprocess/build_quality_groups.py" --root_dir "${OUTPUT_DIR}"
python3 "${ROOT_DIR}/data/preprocess/ordering_new.py" --root_dir "${OUTPUT_DIR}"
