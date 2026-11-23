#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 || ( "$#" -eq 2 && "$2" != --no-install ) ]]; then
    printf 'Usage: bash setup_xpolicylab.sh /path/to/XPolicyLab [--no-install]\n' >&2
    exit 1
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$(command -v python || command -v python3)"
"${python_bin}" "${ROOT_DIR}/policy/RoboTTT/register.py" "$1"
if [[ "${2:-}" != --no-install ]]; then
    "${python_bin}" -m pip install -e "${ROOT_DIR}[isaac-export]"
    bash "$1/policy/RoboTTT/install.sh"
fi
