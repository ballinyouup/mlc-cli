#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

BUILD_VENV="mlc-llm-venv"

conda activate ${BUILD_VENV}

# git clone --recursive https://github.com/mlc-ai/mlc-llm.git
cd mlc-llm2025

mkdir -p build && cd build

python3 ../cmake/gen_cmake_config.py

if [[ "$(uname)" == "Darwin" ]]; then
    NCORES=$(sysctl -n hw.ncpu)
else
    NCORES=$(nproc)
fi

cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . --parallel ${NCORES}

conda deactivate

