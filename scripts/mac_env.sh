#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept environment names as arguments, or use defaults
CLI_VENV="${1:-mlc-cli-venv}"
BUILD_VENV="${2:-mlc-llm-venv}"
CLEAR_EXISTING="${3:-no}"

if [ "$CLEAR_EXISTING" = "yes" ]; then
    echo "Removing existing environments..."
    echo -e "\ny\ny" | conda env remove --name ${CLI_VENV}
    echo -e "\ny\ny" | conda env remove --name ${BUILD_VENV}

    echo "Creating build environment: ${BUILD_VENV}"
    conda create -n ${BUILD_VENV} -c conda-forge --yes \
        "cmake>=3.24" \
        rust \
        git \
        python=3.11 \
        pip \
        git-lfs

    echo "Creating CLI environment: ${CLI_VENV}"
    conda create -n ${CLI_VENV} -c conda-forge --yes \
        "cmake>=3.24" \
        rust \
        git \
        python=3.11 \
        pip \
        git-lfs

    echo "Environments created successfully"
fi