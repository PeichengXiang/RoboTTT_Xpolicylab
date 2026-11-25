#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

mode="${1:-}"
case "$mode" in
    smoke|train) ;;
    *) echo "Usage: bash cloud/run_gr00t_dlc.sh {smoke|train}" >&2; exit 2 ;;
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

"$python" - "$base" "$processor" <<'PY'
from hashlib import sha1, sha256
from pathlib import Path
import re
import sys

import torch
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
models = [re.search(r"(?<![A-Za-z0-9])A(?:100|800)(?![A-Za-z0-9])", name) for name in names]
if (len(names) != 4 or not all(models) or len({match.group() for match in models}) != 1
        or any("MIG" in name.upper() for name in names) or not torch.cuda.is_bf16_supported()):
    raise SystemExit(f"Expected 4 full, same-model NVIDIA A100s or A800s with BF16 support, got {names}")

base, processor = map(Path, sys.argv[1:])
shards = {
    "model-00001-of-00002.safetensors": "8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d",
    "model-00002-of-00002.safetensors": "c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661",
}
for name, expected in shards.items():
    digest = sha256()
    with (base / name).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit(f"Official snapshot SHA-256 mismatch for {name}")

processing_blobs = {
    "tokenizer.json": "c6cc1014128b19d1fc46b1d30a23e3b1d35db421",
    "tokenizer_config.json": "d3d3763207692c78780f4f42d4dadf49a5c8012",
    "preprocessor_config.json": "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
    "video_preprocessor_config.json": "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
    "merges.txt": "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
    "vocab.json": "4783fe10ac3adce15ac8f358ef5462739852c569",
    "chat_template.json": "e49e75bdda104c8de941eb20e0b3765184fb27bc",
    "config.json": "0cd8c646eb55794594b14fefcd1d3acea5cc000f",
}
for name, expected in processing_blobs.items():
    data = (processor / name).read_bytes()
    if sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != expected:
        raise SystemExit(f"Cosmos-compatible public processor hash mismatch for {name}")
print(f"Verified official GR00T, exact processor files, BF16 GPUs {names}, and the NVIDIA processor import", flush=True)
PY

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

case "$mode" in
    smoke) settings=(--smoke --steps 1 --global-batch 4 --context 192 --context-start 192 --context-warmup 0 --save-every 1) ;;
    train) settings=(--steps 50000 --global-batch 64 --save-every 10000) ;;
esac
resume=()
if [[ -n "${ROBOTTT_RESUME:-}" ]]; then
    resume=(--resume "$ROBOTTT_RESUME")
fi
exec "$python" -m torch.distributed.run --standalone --nproc-per-node=4 -m robottt.gr00t_train \
    --data "$data" --base "$base" --processor-assets "$processor" --cache "$cache" --output "$output" \
    --require-a100-a800 "${settings[@]}" "${resume[@]}"
