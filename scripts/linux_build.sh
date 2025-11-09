#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

BUILD_VENV="mlc-chat-venv"

conda activate ${BUILD_VENV}

git clone --recursive https://github.com/mlc-ai/mlc-llm.git && cd mlc-llm

mkdir -p build && cd build

python3 ../cmake/gen_cmake_config.py

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

cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && make -j $(nproc) && cd ..

conda deactivate
