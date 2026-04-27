#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOCA_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SPLIT="${1:-}"
JSON_FILE="${2:-./datasets/MoCA3D}"
CONFIG_PATH="${3:-./configs/MoCA_config.yaml}"

if [[ -z "${SPLIT}" ]]; then
    echo "Usage: bash ./scripts/build_moca_wds.sh <split> [json_file_or_dir] [config_path]" >&2
    exit 1
fi

cd "${MOCA_ROOT}"

python3 ./data/preprocess/extract_features.py \
    --config "${CONFIG_PATH}" \
    --json-file "${JSON_FILE}" \
    --split "${SPLIT}"

python3 ./data/preprocess/converts_to_wds.py \
    --config "${CONFIG_PATH}" \
    --json-file "${JSON_FILE}" \
    --split "${SPLIT}"
