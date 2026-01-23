#!/bin/bash
set -e  # Exit on error

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept parameters
CLI_VENV="${1:-mlc-cli-venv}"
MODEL_URL="${2}"
MODEL_NAME="${3}"
DEVICE="${4:-metal}"
OVERRIDES="${5}"

conda activate "${CLI_VENV}"
mkdir -p models
if [ -z "${MODEL_NAME}" ]; then
    if [ -d "models" ]; then
        MODEL_NAME=$(find -1 models 2>/dev/null | head -n 1)
    fi
fi
MODEL_PATH="models/${MODEL_NAME}"

# Clone model if URL is provided and model doesn't exist
if [ -n "${MODEL_URL}" ]; then
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "Cloning model from HuggingFace..."
        cd models
        git clone "${MODEL_URL}"
        cd "${MODEL_NAME}"
        git lfs pull
        cd ../..
    else
        echo "Model already exists, skipping download..."
    fi
else
    echo "Using local model: ${MODEL_NAME}"
fi

# Run the model
if [ -z "${MODEL_NAME}" ] || [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: MODEL_NAME not provided or model directory not found: ${MODEL_PATH}"
    echo "Available models:"
    ls -1 models 2>/dev/null || true
    conda deactivate
    exit 1
fi
cd "${MODEL_PATH}"
if [ -n "${OVERRIDES}" ]; then
    if command -v mlc_llm >/dev/null 2>&1; then
        MLC_JIT_POLICY=REDO mlc_llm chat . --device "${DEVICE}" --overrides "${OVERRIDES}"
    else
        MLC_JIT_POLICY=REDO python -m mlc_llm chat . --device "${DEVICE}" --overrides "${OVERRIDES}"
    fi
else
    if command -v mlc_llm >/dev/null 2>&1; then
        MLC_JIT_POLICY=REDO mlc_llm chat . --device "${DEVICE}"
    else
        MLC_JIT_POLICY=REDO python -m mlc_llm chat . --device "${DEVICE}"
    fi
fi

conda deactivate
