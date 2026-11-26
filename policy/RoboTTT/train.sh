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

if [[ "${ROBOTTT_RECIPE:-}" == paper_two_stage || "${ROBOTTT_BACKEND:-}" == gr00t_n1_7 ]]; then
    output="${ROBOTTT_RUN_DIR:-${SCRIPT_DIR}/checkpoints/${run_name}}"
    export ROBOTTT_DLC_GPUS="${#gpus[@]}"
    assets="${ROBOTTT_ASSETS_ROOT:-/mnt/workspace/robottt/assets}"
    exec bash "$SCRIPT_DIR/train_gr00t_stages.sh" train "$output" \
        "${ROBOTTT_PYTHON:-python}" -m torch.distributed.run --standalone --nproc-per-node "${#gpus[@]}" \
        -m robottt.gr00t_train --base "${ROBOTTT_GR00T_BASE:-$assets/gr00t-n1.7-3b-373d34e}" \
        --processor-assets "${ROBOTTT_GR00T_PROCESSOR:-$assets/qwen3-vl-2b-processor-8964489}" \
        --data "${ROBOTTT_DATA:-/mnt/workspace/raw}" --cache "${ROBOTTT_GR00T_CACHE:-/mnt/workspace/robottt/cache/gr00t}" \
        --output "$output" --seed "$seed" --global-batch 64 --stride 40 --segment 8 \
        --ttt-variant "${ROBOTTT_TTT_VARIANT:-reference}" --require-a100-a800
fi

exec python -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" -m robottt.ddp_train \
    --data "${ROBOTTT_DATA:-/mnt/workspace/raw}" --output "${SCRIPT_DIR}/checkpoints/${run_name}" \
    --seed "${seed}" --steps "${ROBOTTT_STEPS:-50000}" --global-batch "${ROBOTTT_GLOBAL_BATCH:-64}"
