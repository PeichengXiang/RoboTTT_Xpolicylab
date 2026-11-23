#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 || "$3" != marvin_dual_gripper || "$4" != joint ]]; then
    printf 'Usage: bash train.sh <bench> <checkpoint> marvin_dual_gripper joint <seed> <gpu_ids>\n' >&2
    exit 1
fi
bench_name="$1"
ckpt_name="$2"
env_cfg_type="$3"
action_type="$4"
seed="$5"
gpu_ids="$6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="$(dirname "${XPL_ROOT}")"
export PYTHONPATH="${WORKSPACE}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"

IFS=, read -r -a gpus <<< "${gpu_ids}"
width="$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${WORKSPACE}" "${env_cfg_type}")"
if [[ "${width}" != 16 ]]; then
    printf 'Configured robot does not match the Marvin training data.\n' >&2
    exit 1
fi
run_name="$(python - "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" <<'PY'
import sys
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name

keys = ("bench_name", "ckpt_name", "env_cfg_type", "action_type", "seed")
print(build_run_dir_name(dict(zip(keys, sys.argv[1:], strict=True))))
PY
)"

exec python -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" -m robottt.ddp_train \
    --data "${ROBOTTT_DATA:-/mnt/workspace/raw}" --output "${SCRIPT_DIR}/checkpoints/${run_name}" \
    --seed "${seed}" --steps "${ROBOTTT_STEPS:-50000}" --global-batch "${ROBOTTT_GLOBAL_BATCH:-64}"
