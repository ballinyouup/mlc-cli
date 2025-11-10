#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept CLI environment name as argument
CLI_VENV="${1:-mlc-cli-venv}"

conda activate ${CLI_VENV}

cd mlc-llm/python
pip install --no-deps .

conda deactivate
