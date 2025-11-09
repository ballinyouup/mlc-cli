#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

CLI_VENV="mlc-cli-venv"

conda activate ${CLI_VENV}

# Install TVM first (for CUDA 13.0)
pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu130

# Install mlc_llm as a package
cd mlc-llm/python && pip install .
cd ..

mkdir -p models
cd ../models

git clone https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC
cd Qwen3-1.7B-q4f16_1-MLC

pip install ninja

echo "Setup complete. Running model..."
echo "Run:"
echo "  conda activate ${CLI_VENV}"
echo "  cd models/Qwen3-1.7B-q4f16_1-MLC"
echo "  mlc_llm chat ."

conda deactivate