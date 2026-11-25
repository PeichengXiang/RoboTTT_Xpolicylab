#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

mode="${1:-}"
case "$mode" in
    smoke|pilot|train) ;;
    *) echo "Usage: bash cloud/run_gr00t_dlc.sh {smoke|pilot|train}" >&2; exit 2 ;;
esac

nodes="${ROBOTTT_DLC_NODES:-1}"
case "$nodes" in
    1) gpus_per_node=4; rendezvous=(--standalone) ;;
    2)
        if [[ "${WORLD_SIZE:-}" != 2 || ! "${RANK:-}" =~ ^[01]$ || -z "${MASTER_ADDR:-}" || ! "${MASTER_PORT:-}" =~ ^[0-9]+$ ]]; then
            echo "Two-node launch requires DLC's WORLD_SIZE=2, node RANK, MASTER_ADDR and MASTER_PORT" >&2
            exit 2
        fi
        gpus_per_node=2
        rendezvous=(--nnodes 2 --node-rank "$RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT")
        ;;
    *) echo "ROBOTTT_DLC_NODES must be 1 or 2" >&2; exit 2 ;;
esac

nas="${ROBOTTT_NAS_ROOT:-/mnt/workspace}"
data="${ROBOTTT_DATA_ROOT:-$nas/raw}"
assets="${ROBOTTT_ASSETS_ROOT:-$nas/robottt/assets}"
cache="${ROBOTTT_GR00T_CACHE:-$nas/robottt/cache/gr00t}"
base="$assets/gr00t-n1.7-3b-373d34e"
processor="$assets/qwen3-vl-2b-processor-8964489"
python="${ROBOTTT_PYTHON:-python3}"
: "${ROBOTTT_RUN_DIR:?Set ROBOTTT_RUN_DIR to the persistent NAS run directory}"
nas="$(realpath -e -- "$nas")"
output="$(realpath -m -- "$ROBOTTT_RUN_DIR")"
if [[ "$nas" == / || "$output" != "$nas/"* || ! -d "$data" ]]; then
    echo "Expected raw data and a run directory on the mounted NAS" >&2
    exit 2
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1

"$python" - "$gpus_per_node" <<'PY'
import re
import sys

import torch
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
models = [re.search(r"(?<![A-Za-z0-9])A(?:100|800)(?![A-Za-z0-9])", name) for name in names]
expected = int(sys.argv[1])
if (len(names) != expected or not all(models) or len({match.group() for match in models}) != 1
        or any("MIG" in name.upper() for name in names) or not torch.cuda.is_bf16_supported()):
    raise SystemExit(f"Expected {expected} full, same-model NVIDIA A100s or A800s with BF16 support on this node, got {names}")

print(f"Verified BF16 GPUs {names} and the NVIDIA processor import", flush=True)
PY

"$python" cloud/stage_gr00t_assets.py verify "$assets"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

case "$mode" in
    smoke) settings=(--smoke --steps 3 --global-batch 64 --context 39 --context-start 39 --context-warmup 0 --save-every 3) ;;
    pilot|train)
        settings=(--steps 50000 --global-batch 64 --context 39 --context-start 7 --context-warmup 10000 --save-every 10000)
        if [[ "$mode" == pilot ]]; then
            settings+=(--stop-after 1000)
        fi
        ;;
esac
resume=()
if [[ -n "${ROBOTTT_RESUME:-}" ]]; then
    resume=(--resume "$ROBOTTT_RESUME")
fi
exec "$python" -m torch.distributed.run "${rendezvous[@]}" --nproc-per-node "$gpus_per_node" -m robottt.gr00t_train \
    --data "$data" --base "$base" --processor-assets "$processor" --cache "$cache" --output "$output" \
    --require-a100-a800 --stride 40 --segment 8 "${settings[@]}" "${resume[@]}"
