#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept CLI environment name and install method as arguments
CLI_VENV="${1:-mlc-cli-venv}"
INSTALL_METHOD="${2:-prebuilt}"  # prebuilt or source

conda activate ${CLI_VENV}

if [ "$INSTALL_METHOD" = "prebuilt" ]; then
    echo "Installing pre-built TVM..."
    WHEEL_FILE=$(ls wheels/mlc_ai_nightly_cpu-*.whl 2>/dev/null | head -n 1)
    if [ -z "$WHEEL_FILE" ]; then
        echo "ERROR: No wheel file found in wheels/ directory"
        echo "Current directory: $(pwd)"
        echo "Looking for: wheels/mlc_ai_nightly_cpu-*.whl"
        exit 1
    fi
    echo "Installing wheel: $WHEEL_FILE"
    pip install "$WHEEL_FILE"
else
    echo "Building TVM from source not yet implemented"
    exit 1
fi

conda deactivate