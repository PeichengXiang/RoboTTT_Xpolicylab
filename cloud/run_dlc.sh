#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

mode="${1:-train}"
case "$mode" in
    --help|-h)
        cat <<'HELP'
Usage: bash cloud/run_dlc.sh [check|train]
check: validate all six sets of 50 HDF5 demonstrations, e.g. from CPU DSW.
train: train one DLC worker on 4 A100 or 4 A800 GPUs, global batch size 64,
       50,000 optimizer steps (including any steps in a resumed checkpoint).

ROBOTTT_NAS_ROOT    NAS mount point, default /mnt/workspace.
ROBOTTT_DATA_ROOT   HDF5 root, default ROBOTTT_NAS_ROOT/raw.
ROBOTTT_RUN_DIR     Required output directory for train. Use a persistent NAS path.
ROBOTTT_CACHE_DIR   Worker-local RGB feature cache, default /tmp/robottt-rgb-cache.
ROBOTTT_PYTHON      Python in the DLC image/venv, default python3.
ROBOTTT_RESUME      Optional complete checkpoint file to resume.
HELP
        exit 0
        ;;
    check|train)
        ;;
    *)
        echo "Unknown mode: $mode. Use --help for usage." >&2
        exit 2
        ;;
esac

nas_root="${ROBOTTT_NAS_ROOT:-/mnt/workspace}"
data_root="${ROBOTTT_DATA_ROOT:-$nas_root/raw}"
python="${ROBOTTT_PYTHON:-python3}"
if [[ "$data_root" != /* || ! -d "$data_root" ]]; then
    echo "Expected the mounted HDF5 data directory: $data_root" >&2
    exit 2
fi

if [[ "$mode" == check ]]; then
    exec "$python" -m robottt.ddp_train --data "$data_root" --check-data
fi

: "${ROBOTTT_RUN_DIR:?Set ROBOTTT_RUN_DIR to the persistent output path on NAS}"
if [[ "$nas_root" != /* || ! -d "$nas_root" || "$ROBOTTT_RUN_DIR" != /* ]]; then
    echo "ROBOTTT_NAS_ROOT and ROBOTTT_RUN_DIR must be absolute paths; mount the NAS first." >&2
    exit 2
fi
nas_root="$(realpath -e -- "$nas_root")"
run_dir="$(realpath -m -- "$ROBOTTT_RUN_DIR")"
if [[ "$nas_root" == / || "$run_dir" != "$nas_root/"* ]]; then
    echo "The output directory must be below the NAS mount: $nas_root" >&2
    exit 2
fi
cache_dir="${ROBOTTT_CACHE_DIR:-/tmp/robottt-rgb-cache}"
if [[ "$cache_dir" != /* ]]; then
    echo "ROBOTTT_CACHE_DIR must be an absolute worker-local path." >&2
    exit 2
fi

"$python" - <<'PY'
import re

import h5py
import numpy
import torch

devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
types = [re.search(r"(?<![A-Za-z0-9])A(?:100|800)(?![A-Za-z0-9])", gpu) for gpu in devices]
if (len(devices) != 4 or not all(types) or len({match.group() for match in types}) != 1
        or any("MIG" in name.upper() for name in devices)):
    raise SystemExit(f"Expected one worker with 4 full A100 or 4 full A800 GPUs, found: {devices}")
print(f"Visible GPUs: {devices}; torch={torch.__version__}; h5py={h5py.__version__}; numpy={numpy.__version__}", flush=True)
PY

# Decode/train only after all six synchronized task directories are present.
"$python" -m robottt.ddp_train --data "$data_root" --check-data
resume=()
if [[ -n "${ROBOTTT_RESUME:-}" ]]; then
    resume=(--resume "$ROBOTTT_RESUME")
fi
exec "$python" -m torch.distributed.run --standalone --nproc-per-node=4 -m robottt.ddp_train \
    --data "$data_root" --cache "$cache_dir" --output "$run_dir" \
    --steps 50000 --global-batch 64 --val-per-task 0 --require-a100-a800 "${resume[@]}"
