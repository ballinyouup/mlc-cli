#!/usr/bin/env bash

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"

# Args
CLI_VENV="${1:-mlc-cli-venv}"

if ! conda env list | awk '{print $1}' | grep -qx "${CLI_VENV}"; then
    conda create -n "${CLI_VENV}" -c conda-forge \
        "cmake>=3.24" \
        rust \
        git \
        python=3.13 \
        psutil
fi

conda activate "${CLI_VENV}"

# install TVM FFI (required C extension for TVM)
pip install -e tvm/3rdparty/tvm-ffi

# install TVM Python package (editable install so it finds libs in build/)
cd tvm/python
pip install -e .
cd ../..
