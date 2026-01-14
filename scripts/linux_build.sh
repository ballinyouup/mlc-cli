#!/bin/bash
set -e  # Exit on error

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept build environment name, CUDA flag, and CUDA architecture as arguments
BUILD_VENV="${1:-mlc-llm-venv}"
USE_CUDA="${2:-auto}"
CUDA_ARCH="${3:-86}"

# Check if mlc-llm directory exists
if [ ! -d "mlc-llm" ]; then
    echo "Error: mlc-llm directory not found. Please clone the repository first."
    exit 1
fi

conda activate ${BUILD_VENV}

if [[ "$(uname)" == "Darwin" ]]; then
    NCORES=$(sysctl -n hw.ncpu)
else
    NCORES=$(nproc)
fi
mkdir -p mlc-llm/build
cd mlc-llm/build

# Determine if we should use CUDA
BUILD_WITH_CUDA=false
if [[ "$USE_CUDA" == "y" ]]; then
    if command -v nvcc &> /dev/null; then
        BUILD_WITH_CUDA=true
    else
        echo "Warning: CUDA requested but nvcc not found. Building with CPU-only support..."
    fi
elif [[ "$USE_CUDA" == "auto" ]]; then
    if command -v nvcc &> /dev/null; then
        BUILD_WITH_CUDA=true
    fi
fi

if [[ "$BUILD_WITH_CUDA" == true ]]; then
    echo "Configuring with CUDA support..."
    # Set CUDA environment variables only when CUDA is available
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDACXX=/usr/local/cuda/bin/nvcc
    export CUDA_HOME=/usr/local/cuda
    echo "Using CUDA compute capability: ${CUDA_ARCH}"
    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
else
    echo "Building with CPU-only support..."
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
fi

cmake --build . --parallel ${NCORES}

conda deactivate