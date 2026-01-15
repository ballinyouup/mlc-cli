#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept environment name as argument, or use default
CLI_VENV="${1:-mlc-cli-venv}"

echo "Creating prebuilt environment: ${CLI_VENV}"
conda create -n ${CLI_VENV} -c conda-forge --yes \
    python=3.13 \
    pip \
    git-lfs \
    libstdcxx-ng

echo "Prebuilt environment created successfully"
