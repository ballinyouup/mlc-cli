#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept parameters
BUILD_VENV="${1:-mlc-chat-venv}"
TVM_SOURCE_DIR="${2:-}"
CUDA="${3:-n}"
ROCM="${4:-n}"
VULKAN="${5:-n}"
METAL="${6:-n}"
OPENCL="${7:-n}"

conda activate ${BUILD_VENV}

cd mlc-llm/build

# Generate answers for gen_cmake_config.py
# The script asks for: TVM_SOURCE_DIR, CUDA, CUTLASS, CUBLAS, ROCM, Vulkan, Metal, OpenCL
# If CUDA is enabled, also enable CUTLASS and CUBLAS by default
CUTLASS="${CUDA}"
CUBLAS="${CUDA}"

# Build the answers string with explicit newlines
# Add extra newlines at the end to handle any additional prompts
printf "%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n\n\n" \
    "${TVM_SOURCE_DIR}" \
    "${CUDA}" \
    "${CUTLASS}" \
    "${CUBLAS}" \
    "${ROCM}" \
    "${VULKAN}" \
    "${METAL}" \
    "${OPENCL}" | python3 ../cmake/gen_cmake_config.py

conda deactivate
