#!/bin/bash
set -e  # Exit on error

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept CLI environment name as argument
CLI_VENV="${1:-mlc-cli-venv}"

# Check if mlc-llm directory exists
if [ ! -d "mlc-llm" ]; then
    echo "Error: mlc-llm directory not found. Please clone the repository first."
    exit 1
fi

conda activate ${LLM_VENV}

# Clean up any existing library files in the package directory to avoid SameFileError
rm -f mlc-llm/python/mlc_llm/libmlc_llm.so
rm -f mlc-llm/python/mlc_llm/libmlc_llm_module.so

# First install MLC Python package
cd mlc-llm/python
pip install -e .
cd ../..

# Then install the TVM wheel (required for running models)
WHEEL_FILE=$(ls wheels/mlc_ai_nightly_cu130-*.whl 2>/dev/null | head -n 1)
if [ -n "$WHEEL_FILE" ]; then
    echo "Installing TVM wheel: $WHEEL_FILE"
    pip install "$WHEEL_FILE"
else
    echo "WARNING: No TVM wheel found in wheels/ directory"
    echo "You may need to install TVM manually"
fi

conda deactivate
