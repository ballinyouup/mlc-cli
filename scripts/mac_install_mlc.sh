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

# install MLC Python package
cd mlc-llm/python

if [ -f requirements.txt ] && grep -q '^flashinfer-python==0\.4\.0$' requirements.txt; then
  # remove flash infer for mac
  sed -i '' -e '/^flashinfer-python==0\.4\.0$/d' requirements.txt
fi

pip install -e .
cd ../..
