#!/usr/bin/env bash
set -euo pipefail

bench_name="$1"
task_name="$2"
env_cfg_type="$4"
seed="$6"
env_gpu_id="$7"
eval_env="$8"
additional_info="$9"
port="${10}"
host="${11:-localhost}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="$(dirname "${XPL_ROOT}")"
UTILS_DIR="${XPL_ROOT}/utils"
source "${UTILS_DIR}/resolve_eval_env_type.sh"

if [[ "$(resolve_eval_env_type)" == debug ]]; then
    source "${SCRIPT_DIR}/python_environment.sh"
    python_bin="$(robottt_python "${eval_env}")"
    export PYTHONPATH="${WORKSPACE}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    exec "${python_bin}" "${UTILS_DIR}/debug_env_client.py" \
        --bench_name "${bench_name}" --task_name "${task_name}" --env_cfg_type "${env_cfg_type}" \
        --policy_name RoboTTT --protocol ws --host "${host}" --port "${port}" \
        --eval_episode_num "${DEBUG_EVAL_EPISODES:-10}" --eval_batch "${DEBUG_EVAL_BATCH:-0}"
fi

exec bash "${UTILS_DIR}/setup_env_client.sh" "${UTILS_DIR}" "${SCRIPT_DIR}/deploy.yml" "${eval_env}" \
    "${port}" "${bench_name}" "${task_name}" "${env_cfg_type}" RoboTTT "${additional_info}" \
    "${WORKSPACE}" "${seed}" "${env_gpu_id}" "${host}"
