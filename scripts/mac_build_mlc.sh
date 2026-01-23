#!/usr/bin/env bash
set -e  # Exit on error

# Args
BUILD_VENV="${1:-mlc-build-venv}"
CUDA="${2:-n}"
ROCM="${3:-n}"
VULKAN="${4:-n}"
METAL="${5:-y}"
OPEN_CL="${6:-n}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"
TVM_SOURCE_DIR="${REPO_ROOT}/tvm"

source "$(conda info --base)/etc/profile.d/conda.sh"

# create the conda environment with build dependency
conda create -y -n "${BUILD_VENV}" -c conda-forge \
    "cmake>=3.24" \
    rust \
    git \
    zstd \
    python=3.13
# enter the build environment
conda activate "${BUILD_VENV}"

# Set library path for cmake to find zstd
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"

# clone from GitHub (or use existing)
if [ ! -d "mlc-llm" ]; then
    git clone --recursive https://github.com/mlc-ai/mlc-llm.git
fi
cd mlc-llm/

# create build directory
mkdir -p build && cd build

# generate build configuration
printf "%s\n%s\n%s\n%s\n%s\n%s\n\n\n" \
    "${TVM_SOURCE_DIR}" \
    "${CUDA}" \
    "${ROCM}" \
    "${VULKAN}" \
    "${METAL}" \
    "${OPEN_CL}" \
    | python3 ../cmake/gen_cmake_config.py

# build mlc_llm libraries
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && make -j4 && cd ..

# Build wheel and copy to wheels directory
mkdir -p "${WHEELS_DIR}"

cd python
pip install build
python -m build --wheel --outdir "${WHEELS_DIR}"
cd ..

echo "MLC-LLM wheel created in ${WHEELS_DIR}"

conda deactivate

