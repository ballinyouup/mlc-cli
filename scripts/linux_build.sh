#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept build environment name as argument
BUILD_VENV="${1:-mlc-chat-venv}"

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

cd mlc-llm/build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . --parallel ${NCORES}

conda deactivate