#!/bin/bash
set -e  # Exit on error

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept build environment name as argument
BUILD_VENV="${1:-mlc-chat-venv}"

# Check if mlc-llm directory exists
if [ ! -d "mlc-llm" ]; then
    echo "Error: mlc-llm directory not found. Please clone the repository first."
    exit 1
fi

conda activate ${BUILD_VENV}

# Set CUDA environment variables
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
export CUDA_HOME=/usr/local/cuda-13.0

if [[ "$(uname)" == "Darwin" ]]; then
    NCORES=$(sysctl -n hw.ncpu)
else
    NCORES=$(nproc)
fi
mkdir -p mlc-llm2025/build
cd mlc-llm2025/build

# Check if CUDA is available and set architecture
if command -v nvcc &> /dev/null; then
    echo "CUDA detected, configuring with CUDA support..."
    # For RTX 3060 and similar GPUs (Ampere architecture)
    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
else
    echo "CUDA not detected, building without CUDA support..."
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
fi

cmake --build . --parallel ${NCORES}

conda deactivate