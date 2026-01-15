#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept environment name and GPU type as arguments
CLI_VENV="${1:-mlc-cli-venv}"
GPU_TYPE="${2:-cpu}"

conda activate ${CLI_VENV}

echo "Installing MLC-LLM prebuilt packages for ${GPU_TYPE}..."

case ${GPU_TYPE} in
    "cuda128")
        python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu128 mlc-ai-nightly-cu128
        ;;
    "cuda130")
        python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu130 mlc-ai-nightly-cu130
        ;;
    "rocm61")
        python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-rocm61 mlc-ai-nightly-rocm61
        ;;
    "rocm62")
        python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-rocm62 mlc-ai-nightly-rocm62
        ;;
    *)
        python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
        ;;
esac

echo "Verifying installation..."
python -c "import mlc_llm; print('MLC-LLM installed at:', mlc_llm.__file__)"

conda deactivate

echo "MLC-LLM prebuilt packages installed successfully"
