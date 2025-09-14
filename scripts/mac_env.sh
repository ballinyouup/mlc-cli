#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

 CLI_VENV="mlc-cli-venv"
 BUILD_VENV="mlc-llm-venv"

echo  -e "\ny\ny" | conda env remove --name ${CLI_VENV}
echo  -e "\ny\ny" | conda env remove --name ${BUILD_VENV}

conda create -n ${BUILD_VENV} -c conda-forge \
    "cmake>=3.24" \
    rust \
    git \
    python=3.11 \
    git-lfs \
#    pip
#    sentencepiece


  conda create -n ${CLI_VENV} -c conda-forge \
    "cmake>=3.24" \
    rust \
    git \
    python=3.11 \
    git-lfs \
#    pip
#    sentencepiece
#    git-lfs