#!/usr/bin/env bash
set -euo pipefail

mode="$1"
output="$2"
shift 2
launch=("$@")
total=50000
boundary=30000
save=10000
pre_context=(--context 39 --context-start 7 --context-warmup 10000)
post_context=(--context 25 --context-start 25 --context-warmup 0)
extra=()
if [[ "$mode" == smoke ]]; then
    total=4
    boundary=2
    save=2
    pre_context=(--context 39 --context-start 39 --context-warmup 0)
    extra=(--smoke)
elif [[ "$mode" == pilot ]]; then
    extra=(--stop-after 1000)
fi
launch+=(--steps "$total" --phase-boundary "$boundary" --save-every "$save" "${extra[@]}")
transition=$(printf '%s/step_%06d.pt' "$output" "$boundary")
completed=0
resume=()
if [[ -n "${ROBOTTT_RESUME:-}" ]]; then
    name=$(basename "$(realpath -e "$ROBOTTT_RESUME")")
    [[ "$name" =~ ^step_([0-9]{6})\.pt$ ]] || { echo "Invalid two-stage resume checkpoint" >&2; exit 2; }
    completed=$((10#${BASH_REMATCH[1]}))
    resume=(--resume "$ROBOTTT_RESUME")
fi
if (( completed < boundary )); then
    "${launch[@]}" --stage pretrain "${pre_context[@]}" "${resume[@]}"
fi
if [[ "$mode" == pilot ]]; then exit 0; fi
if (( completed <= boundary )); then
    resume=(--init-from "$transition")
fi
exec "${launch[@]}" --stage posttrain "${post_context[@]}" "${resume[@]}"
