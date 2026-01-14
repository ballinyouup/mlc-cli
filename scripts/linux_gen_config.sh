#!/bin/bash
set -e  # Exit on error

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept parameters
BUILD_VENV="${1:-mlc-llm-venv}"
TVM_SOURCE_DIR="${2:-}"
CUDA="${3:-n}"
ROCM="${4:-n}"
VULKAN="${5:-n}"
OPENCL="${6:-n}"
USE_PREBUILT_TVM="${7:-n}"
USE_PREBUILT_MLC="${8:-n}"

# Check if mlc-llm directory exists
if [ ! -d "mlc-llm" ]; then
    echo "Error: mlc-llm directory not found. Please clone the repository first."
    exit 1
fi

conda activate ${BUILD_VENV}

mkdir -p mlc-llm/build
cd mlc-llm/build

# Generate answers for gen_cmake_config.py
# The script asks for: TVM_SOURCE_DIR, CUDA, CUTLASS, CUBLAS, ROCM, Vulkan, Metal, OpenCL, FlashInfer, CUDA Compute Capability
# If CUDA is enabled, also enable CUTLASS, CUBLAS, and FlashInfer by default
CUTLASS="n"
CUBLAS="n"
FLASHINFER="n"
# RTX 3060 Ti has compute capability 8.6, which is entered as 86
CUDA_COMPUTE_CAP="86"

# Build the answers string with explicit newlines
# Add extra newlines at the end to handle any additional prompts
printf "%s\n%s\n%s\n%s\n%s\n%s\nn\n%s\n%s\n\n\n" \
    "${TVM_SOURCE_DIR}" \
    "${CUDA}" \
    "${CUTLASS}" \
    "${CUBLAS}" \
    "${ROCM}" \
    "${VULKAN}" \
    "${OPENCL}" \
    "${FLASHINFER}" | python3 ../cmake/gen_cmake_config.py

conda deactivate
