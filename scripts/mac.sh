#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

CLI_VENV="mlc-cli-venv"
BUILD_VENV="mlc-llm-venv"

echo  -e "\ny\ny" | conda env remove --name ${CLI_VENV}
echo  -e "\ny\ny" | conda env remove --name ${BUILD_VENV}

conda create -n ${CLI_VENV} -c conda-forge --yes \
    "cmake=3.29" \
    rust \
    git \
    python=3.11 \
    pip \
    sentencepiece \
    git-lfs

conda create -n ${BUILD_VENV} -c conda-forge --yes \
    "cmake=3.29" \
    rust \
    git \
    python=3.11 \
    pip \
    sentencepiece \
    git-lfs

conda run -n ${CLI_VENV} pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu

if [ ! -d "mlc-llm" ]; then
    git clone --recursive https://github.com/mlc-ai/mlc-llm.git
fi
cd mlc-llm

mkdir -p build && cd build

(conda activate ${BUILD_VENV} && echo -e "\nn\nn\nn\ny\nn\nn\nn\nn\nn" | python3 ../cmake/gen_cmake_config.py)

if [[ "$(uname)" == "Darwin" ]]; then
    NCORES=$(sysctl -n hw.ncpu)
else
    NCORES=$(nproc)
fi

(conda activate ${BUILD_VENV} && cmake ..)
(conda activate ${BUILD_VENV} && cmake --build . --parallel ${NCORES})

cd ..

(conda activate ${CLI_VENV} && cd python && pip install .)

mkdir -p models
cd models

if [ ! -d "Qwen3-1.7B-q4f16_1-MLC" ]; then
    conda run -n ${CLI_VENV} git clone https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC
fi
cd Qwen3-1.7B-q4f16_1-MLC
conda run -n ${CLI_VENV} git lfs pull

echo "Setup complete. Running model..."
echo "Run:"
echo "  conda activate ${CLI_VENV}"
echo "  cd mlc-llm/models/Qwen3-1.7B-q4f16_1-MLC"
echo "  mlc_llm chat . --device metal"