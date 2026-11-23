#!/usr/bin/env bash
# Select a Python interpreter without mutating the launcher's current environment.
robottt_python() {
    local environment="$1"
    if [[ -x "${environment}/bin/python" ]]; then
        printf '%s\n' "${environment}/bin/python"
    elif [[ "${environment}" == current ]]; then
        command -v python
    else
        (
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate "${environment}"
            command -v python
        )
    fi
}
