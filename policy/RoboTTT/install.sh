#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ ! -f "${XPL_ROOT}/setup_policy_server.py" ]]; then
    printf 'Install the adapter into an XPolicyLab checkout with setup_xpolicylab.sh first.\n' >&2
    exit 1
fi
python -m pip install -e "${XPL_ROOT}"
python -c 'import robottt'
