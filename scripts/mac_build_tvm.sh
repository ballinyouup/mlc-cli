#!/usr/bin/env bash
set -e  # Exit on error

# Args
BUILD_VENV="${1:-tvm-build-venv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n ${BUILD_VENV} -c conda-forge \
    "llvmdev=17" \
    "cmake>=3.24" \
    git \
    zstd \
    python=3.13

conda activate ${BUILD_VENV}

# Set library path for cmake to find zstd
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"

# clone from GitHub (or use existing)
if [ ! -d "tvm" ]; then
    git clone --recursive https://github.com/apache/tvm.git
fi
cd tvm
# create the build directory
rm -rf build && mkdir build && cd build
# specify build requirements in `config.cmake`
cp ../cmake/config.cmake .

# controls default compilation flags (use sed to replace existing values)
sed -i '' 's/set(CMAKE_BUILD_TYPE .*/set(CMAKE_BUILD_TYPE Release)/' config.cmake
# LLVM is a must dependency
sed -i '' 's|set(USE_LLVM .*)|set(USE_LLVM "llvm-config --ignore-libllvm --link-static")|' config.cmake
sed -i '' 's/set(HIDE_PRIVATE_SYMBOLS .*/set(HIDE_PRIVATE_SYMBOLS ON)/' config.cmake
# GPU SDKs - enable Metal for macOS
sed -i '' 's/set(USE_CUDA .*/set(USE_CUDA OFF)/' config.cmake
sed -i '' 's/set(USE_ROCM .*/set(USE_ROCM OFF)/' config.cmake
sed -i '' 's/set(USE_METAL .*/set(USE_METAL ON)/' config.cmake
sed -i '' 's/set(USE_VULKAN .*/set(USE_VULKAN OFF)/' config.cmake
sed -i '' 's/set(USE_OPENCL .*/set(USE_OPENCL OFF)/' config.cmake

cmake .. && make -j4 && cd ..

# Build wheels and copy to wheels directory
mkdir -p "${WHEELS_DIR}"

# Build TVM wheel
cd python
pip install build
python -m build --wheel --outdir "${WHEELS_DIR}"
cd ..

echo "TVM wheels created in ${WHEELS_DIR}"

