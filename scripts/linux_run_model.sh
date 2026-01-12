#!/bin/bash
set -e  # Exit on error

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept parameters
CLI_VENV="${1:-mlc-cli-venv}"
MODEL_URL="${2}"
MODEL_NAME="${3}"

# Set CUDA environment variables
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
export CUDA_HOME=/usr/local/cuda-13.0

conda activate ${CLI_VENV}
mkdir -p models
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
cd ${MODEL_PATH}
MLC_JIT_POLICY=REDO mlc_llm chat . --device cuda

conda deactivate
