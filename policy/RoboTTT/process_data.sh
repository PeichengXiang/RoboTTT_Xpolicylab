#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$3" != marvin_dual_gripper || "$4" != joint ]]; then
    printf 'Usage: ROBOTTT_DATA=/path/to/raw bash process_data.sh <bench> <checkpoint> marvin_dual_gripper joint [50]\n' >&2
    exit 1
fi

raw_path="${ROBOTTT_DATA:-/mnt/workspace/raw}"
if [[ "${5:-50}" != 50 ]]; then
    printf 'The six-task training corpus expects 50 expert episodes per task.\n' >&2
    exit 1
fi
exec python -m robottt.ddp_train --data "${raw_path}" --check-data
