#!/bin/bash
set -e  # Exit on error

source "$(conda info --base)/etc/profile.d/conda.sh"

# Args
BUILD_VENV="${1:-mlc-llm-venv}"
CUDA="${2:-y}"
CUTLASS="${3:-n}"
CUBLAS="${4:-n}"
ROCM="${5:-n}"
VULKAN="${6:-n}"
OPENCL="${7:-n}"
FLASHINFER="${8:-n}"
CUDA_ARCH="${9:-86}"

# Variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"
TVM_SOURCE_DIR="${REPO_ROOT}/tvm"

echo "Creating build environment: ${BUILD_VENV}"
conda create -n ${BUILD_VENV} -c conda-forge --yes \
      "cmake>=3.24" \
      rust \
      git \
      python=3.13 \
      pip \
      git-lfs

echo "${BUILD_VENV} environment created successfully"

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

# Generate CMake config
printf "%s\n%s\n%s\n%s\n%s\n%s\nn\n%s\n%s\n\n\n" \
    "${TVM_SOURCE_DIR}" \
    "${CUDA}" \
    "${CUTLASS}" \
    "${CUBLAS}" \
    "${ROCM}" \
    "${VULKAN}" \
    "${OPENCL}" \
    "${FLASHINFER}" | python3 ../cmake/gen_cmake_config.py

if [[ "$CUDA" == true ]]; then
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

# Build wheel and copy to wheels directory
mkdir -p "${WHEELS_DIR}"

cd python
pip install build
python -m build --wheel --outdir "${WHEELS_DIR}"
cd ..

echo "MLC-LLM wheel created in ${WHEELS_DIR}"

conda deactivate