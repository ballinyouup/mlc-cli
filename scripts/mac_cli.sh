#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

CLI_VENV="mlc-cli-venv"
BUILD_VENV="mlc-llm-venv"

conda activate ${CLI_VENV}

# install mlc as a package
# if mlc_llm is not recognized, comment out flash infer in python/requirements.txt
cd mlc-llm/python && pip install .
cd ..
mkdir -p models
cd ../models

git clone https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC
cd Qwen3-1.7B-q4f16_1-MLC

# install TVM
pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu
pip install ninja
echo "Setup complete. Running model..."
echo "Run:"
echo "  conda activate ${CLI_VENV}"
echo "  cd models/Qwen3-1.7B-q4f16_1-MLC"
echo "  MLC_JIT_POLICY=REDO mlc_llm chat . --device metal"

conda deactivate

# note: for different repo's,
# please use submodule update --init --recursive --remote to fetch 3rdparty libs