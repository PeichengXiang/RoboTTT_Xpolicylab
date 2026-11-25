#!/usr/bin/env bash
set -euo pipefail

nas="${ROBOTTT_NAS_ROOT:-/mnt/workspace}"
if [[ "$nas" != /* || ! -d "$nas" ]]; then
    echo "Expected an absolute, mounted ROBOTTT_NAS_ROOT directory: $nas" >&2
    exit 2
fi
nas="$(realpath -e -- "$nas")"
if [[ "$nas" == / || "$(uname -m)" != x86_64 || "$(uname -s)" != Linux ]]; then
    echo "Use a Linux x86_64 DSW and a NAS mount below /." >&2
    exit 2
fi

revision=708617f5716c63fc0a327391492ea2b91977ae90
source_url=https://github.com/NVIDIA/Isaac-GR00T.git
upstream="$nas/robottt/vendor/Isaac-GR00T-$revision"
runtime="$nas/robottt/runtime/gr00t-cu128"
bootstrap_python="${ROBOTTT_BOOTSTRAP_PYTHON:-python3}"
uvbin="$runtime/bootstrap/bin/uv"
mkdir -p "$runtime" "$upstream"
exec 9>"$runtime/.prepare.lock"
if ! flock -n 9; then
    echo "Another process is preparing $runtime" >&2
    exit 1
fi

if [[ ! -x "$uvbin" ]]; then
    uv_source=uv==0.10.12
    if [[ -n "${ROBOTTT_UV_WHEEL:-}" ]]; then
        uv_source="$(realpath -e -- "$ROBOTTT_UV_WHEEL")"
    fi
    "$bootstrap_python" -m pip install --disable-pip-version-check --no-deps \
        --upgrade --target "$runtime/bootstrap" "$uv_source"
fi
if ! "$uvbin" --version | grep -Eq '^uv 0\.10\.12([[:space:]]|$)'; then
    echo "The shared bootstrap must contain uv 0.10.12: $uvbin" >&2
    exit 1
fi

if [[ ! -d "$upstream/.git" ]]; then
    git init -q "$upstream"
fi
if ! git -C "$upstream" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$upstream" fetch --depth=1 "$source_url" "$revision"
fi
GIT_LFS_SKIP_SMUDGE=1 git -C "$upstream" checkout --quiet --detach "$revision"
if [[ "$(git -C "$upstream" rev-parse HEAD)" != "$revision" ||
      -n "$(git -C "$upstream" status --porcelain --untracked-files=normal)" ]]; then
    echo "Expected the clean, pinned NVIDIA source at $upstream" >&2
    exit 1
fi

export UV_PYTHON_INSTALL_DIR="$runtime/python"
export UV_PROJECT_ENVIRONMENT="$runtime/venv"
export UV_CACHE_DIR="$runtime/cache"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
"$uvbin" python install 3.12.13 --no-bin
"$uvbin" sync --project "$upstream" --frozen --managed-python --python 3.12.13 \
    --no-dev --no-editable --no-install-package flash-attn \
    --no-install-package tensorrt-cu12 --no-install-package tensorrt-cu12-bindings \
    --no-install-package tensorrt-cu12-libs --no-install-package deepspeed \
    --no-install-package torchcodec

python="$UV_PROJECT_ENVIRONMENT/bin/python"
h5py_source=h5py==3.13.0
if [[ -n "${ROBOTTT_H5PY_WHEEL:-}" ]]; then
    h5py_source="$(realpath -e -- "$ROBOTTT_H5PY_WHEEL")"
fi
"$uvbin" pip install --python "$python" --no-deps "$h5py_source"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
    "$python" -I - "$runtime" <<'PY'
import sys
from pathlib import Path

import diffusers
import h5py
import torch
import transformers
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

runtime = Path(sys.argv[1]).resolve()
if (sys.version_info[:3] != (3, 12, 13)
        or Path(sys.prefix).resolve() != runtime / "venv"
        or not Path(sys.base_prefix).resolve().is_relative_to(runtime / "python")):
    raise SystemExit(f"Unexpected Python environment: {sys.executable}, {sys.base_prefix}")
expected = ("2.9.0+cu128", "12.8", "4.57.3", "0.35.1", "3.13.0")
actual = (torch.__version__, torch.version.cuda, transformers.__version__,
          diffusers.__version__, h5py.__version__)
if actual != expected:
    raise SystemExit(f"Unexpected training dependencies: expected {expected}, got {actual}")
print(f"CPU import preflight passed: Python {sys.version.split()[0]}, versions {actual}")
print(f"ROBOTTT_PYTHON={sys.executable}")
PY
