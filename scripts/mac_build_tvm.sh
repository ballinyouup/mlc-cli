#!/usr/bin/env bash
set -e  # Exit on error

# Load central version/dependency configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/versions.sh"

# Args
BUILD_VENV="${1:-tvm-build-venv}"
TVM_SOURCE="${2:-bundled}"  # bundled or custom
BUILD_WHEELS="${3:-y}"
FORCE_CLONE="${4:-n}"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n ${BUILD_VENV} -c "${CONDA_CHANNEL}" \
    "llvmdev=19" \
    "cmake>=${CMAKE_MIN_VERSION}" \
    git \
    zstd \
    "${PYTHON_PACKAGE_SPEC}" \
    "${PYTHON_ABI_SPEC}"

conda activate ${BUILD_VENV}

# Set library path for cmake to find zstd
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"

# Determine TVM directory based on source selection
if [ "${TVM_SOURCE}" = "custom" ]; then
    TVM_DIR="${REPO_ROOT}/tvm"
    echo "Using custom TVM from ${TVM_DIR}"
    if [ ! -d "${TVM_DIR}" ]; then
        echo "Error: Custom TVM directory not found at ${TVM_DIR}"
        echo "Please clone TVM to ${TVM_DIR} or select bundled TVM option"
        exit 1
    fi
elif [ "${TVM_SOURCE}" = "relax" ]; then
    TVM_DIR="${REPO_ROOT}/tvm"
    echo "Using mlc-ai/relax (mlc branch) at ${TVM_DIR}"
    if [ "${FORCE_CLONE}" = "y" ] && [ -d "${TVM_DIR}" ]; then
        echo "Force re-clone: removing existing ${TVM_DIR}..."
        rm -rf "${TVM_DIR}"
    fi
    if [ ! -d "${TVM_DIR}" ]; then
        echo "Cloning ${TVM_REPO} ref=${TVM_REF}..."
        git clone "${TVM_REPO}" "${TVM_DIR}"
        git -C "${TVM_DIR}" checkout "${TVM_REF}"
        git -C "${TVM_DIR}" submodule update --init --recursive
    elif [ "$(git -C "${TVM_DIR}" rev-parse HEAD 2>/dev/null)" != "$(git -C "${TVM_DIR}" rev-parse "${TVM_REF}" 2>/dev/null || echo 'unknown')" ]; then
        current_tvm_head="$(git -C "${TVM_DIR}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
        echo "Current TVM HEAD is ${current_tvm_head:0:8}, expected ${TVM_REF}"
        echo "Switching TVM to ${TVM_REF}..."
        git -C "${TVM_DIR}" remote set-url origin "${TVM_REPO}"
        git -C "${TVM_DIR}" fetch origin
        git -C "${TVM_DIR}" checkout "${TVM_REF}"
        git -C "${TVM_DIR}" submodule update --init --recursive
    else
        echo "TVM is already at ${TVM_REF}."
    fi
else
    # Clone mlc-llm if it doesn't exist (for bundled TVM)
    MLC_LLM_DIR="${REPO_ROOT}/mlc-llm"
    if [ "${FORCE_CLONE}" = "y" ] && [ -d "${MLC_LLM_DIR}" ]; then
        echo "Force re-clone: removing existing ${MLC_LLM_DIR}..."
        rm -rf "${MLC_LLM_DIR}"
    fi
    if [ ! -d "${MLC_LLM_DIR}" ]; then
        echo "mlc-llm not found, cloning from ${MLC_LLM_REPO}..."
        git clone --recursive "${MLC_LLM_REPO}" "${MLC_LLM_DIR}"
    fi
    
    TVM_DIR="${REPO_ROOT}/mlc-llm/3rdparty/tvm"
    echo "Using bundled TVM from ${TVM_DIR}"
    if [ ! -d "${TVM_DIR}" ]; then
        echo "Error: Bundled TVM directory not found at ${TVM_DIR}"
        echo "Initializing submodules..."
        git -C "${MLC_LLM_DIR}" submodule update --init --recursive
    fi
fi

cd "${TVM_DIR}"
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

cmake .. && make -j4
cd ..

if [ "${BUILD_WHEELS}" = "y" ]; then
    # Build wheels and copy to wheels directory
    mkdir -p "${WHEELS_DIR}"

    # Clean CMake cache and Makefiles to avoid Make/Ninja conflict
    # but keep the compiled libraries in build/
    cd build
    rm -f Makefile CMakeCache.txt cmake_install.cmake
    rm -rf CMakeFiles
    cd ..

    # Build TVM wheel from the tvm root directory (where pyproject.toml is)
    python -m pip install build
    python -m build --wheel --outdir "${WHEELS_DIR}"

    echo "TVM wheels created in ${WHEELS_DIR}"
else
    echo "Skipping TVM wheel build."
fi

