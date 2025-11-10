#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept parameters
CLI_VENV="${1:-mlc-cli-venv}"
MODEL_URL="${2}"
MODEL_NAME="${3}"

conda activate ${CLI_VENV}

MODEL_PATH="mlc-llm/models/${MODEL_NAME}"

# Clone model if it doesn't exist
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Cloning model..."
    cd mlc-llm/models
    git clone ${MODEL_URL}
    cd ${MODEL_NAME}
    git lfs pull
    cd ../../..
fi

# Run the model
cd ${MODEL_PATH}
mlc_llm chat . --device cuda

conda deactivate
