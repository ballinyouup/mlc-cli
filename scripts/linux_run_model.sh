#!/bin/bash
set -e  # Exit on error

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept parameters
CLI_VENV="${1:-mlc-cli-venv}"
MODEL_URL="${2}"
MODEL_NAME="${3}"

# Check if mlc-llm directory exists
if [ ! -d "mlc-llm2025" ]; then
    echo "Error: mlc-llm2025 directory not found. Please clone the repository first."
    exit 1
fi

conda activate ${CLI_VENV}

MODEL_PATH="mlc-llm2025/models/${MODEL_NAME}"

# Clone model if it doesn't exist
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Cloning model..."
    cd mlc-llm2025/models
    git clone ${MODEL_URL}
    cd ${MODEL_NAME}
    git lfs pull
    cd ../../..
fi

# Run the model
cd ${MODEL_PATH}
MLC_JIT_POLICY=REDO mlc_llm chat . --device cuda

conda deactivate
