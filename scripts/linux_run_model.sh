#!/bin/bash
set -e  # Exit on error

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept parameters
CLI_VENV="${1:-mlc-cli-venv}"
MODEL_URL="${2}"
MODEL_NAME="${3}"
DEVICE="${4:-cuda}"

# Set CUDA environment variables
DEVICE_FLAG=""
if [ "${DEVICE}" = "cuda" ]; then
    export PATH=/usr/local/cuda-13.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
    export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
    export CUDA_HOME=/usr/local/cuda-13.0
    DEVICE_FLAG="--device cuda"
elif [ "${DEVICE}" = "cpu" ] || [ "${DEVICE}" = "none" ] || [ -z "${DEVICE}" ]; then
    DEVICE_FLAG=""
else
    DEVICE_FLAG="--device ${DEVICE}"
fi

conda activate ${CLI_VENV}
mkdir -p models
if [ -z "${MODEL_NAME}" ]; then
    if [ -d "models" ]; then
        MODEL_NAME=$(ls -1 models 2>/dev/null | head -n 1)
    fi
fi
MODEL_PATH="models/${MODEL_NAME}"

# Clone model if URL is provided and model doesn't exist
if [ -n "${MODEL_URL}" ]; then
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "Cloning model from HuggingFace..."
        cd models
        git clone ${MODEL_URL}
        cd ${MODEL_NAME}
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
cd ${MODEL_PATH}
if command -v mlc_llm >/dev/null 2>&1; then
    MLC_JIT_POLICY=REDO mlc_llm chat . ${DEVICE_FLAG}
else
    MLC_JIT_POLICY=REDO python -m mlc_llm chat . ${DEVICE_FLAG}
fi

conda deactivate
