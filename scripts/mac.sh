#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

VENV="mlc-llm-venv"

echo  -e "\ny\ny" | conda env remove --name ${VENV}

conda create -n ${VENV} -c conda-forge --yes \
    cmake=3.29 \
    rust \
    git \
    python=3.10 \
    pip=25.1 \
    sentencepiece \
    git-lfs \

conda activate ${VENV}

if [ ! -d "mlc-llm" ]; then
    git clone --recursive https://github.com/mlc-ai/mlc-llm.git
fi
cd mlc-llm

echo "Patching outdated CMake file..."
sed -i.bak 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.12)/' 3rdparty/tokenizers-cpp/msgpack/CMakeLists.txt


mkdir -p build && cd build

echo -e "\nn\nn\nn\ny\nn\nn\nn\nn\nn" | python3 ../cmake/gen_cmake_config.py

if [[ "$(uname)" == "Darwin" ]]; then
    # export NCORES=$(sysctl -n hw.ncpu)
    NCORES=$(sysctl -n hw.ncpu)
else
    NCORES=$(nproc)
fi

cmake ..
cmake --build . --parallel ${NCORES}

cd ..

conda deactivate

# install mlc as a package
# if mlc_llm is not recognized, comment out flash infer in python/requirements.txt
if [ -f "python/requirements.txt" ]; then
    sed -i.bak 's/^flashinfer/# flashinfer/' python/requirements.txt
    echo "Commented out flashinfer in requirements.txt"
fi
cd python && python -m pip install .
cd ..
mkdir -p models

if [ ! -d "Qwen3-1.7B-q4f16_1-MLC" ]; then
    git clone https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC
fi
cd Qwen3-1.7B-q4f16_1-MLC
git lfs pull

# install TVM
pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu

echo "Setup complete. Running model..."
echo "Run:"
echo "  conda activate ${CLI_VENV}"
echo "  cd mlc-llm/models/Qwen3-1.7B-q4f16_1-MLC"
echo "  mlc_llm chat . --device metal"

conda deactivate
## Notes
# python 3.10
# possibly remove model cache from .cache folder.
# build in parallel or it'll take forever
# split into doing build env first then cli env second
# comment out FlashInfer in requirements.txt on mac
