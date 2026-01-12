#!/bin/bash
set -e  # Exit on error

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept parameters
CLI_VENV="${1:-mlc-cli-venv}"
MODEL_URL="${2}"
MODEL_NAME="${3}"

conda activate ${CLI_VENV}
mkdir -p models
MODEL_PATH="models/${MODEL_NAME}"

# Clone model if it doesn't exist
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Cloning model..."
    cd models
    git clone ${MODEL_URL}
    cd ${MODEL_NAME}
    git lfs pull
    cd ../..
fi

# Run the model
cd ${MODEL_PATH}
MLC_JIT_POLICY=REDO mlc_llm chat . --device metal

conda deactivate
