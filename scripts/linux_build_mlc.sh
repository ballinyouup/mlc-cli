#!/bin/bash
# =============================================================================
# MLC-LLM Build Script for Linux
# =============================================================================
# This script builds MLC-LLM from source with configurable options for various
# GPU backends (CUDA, ROCm, Vulkan, OpenCL) and optimization features.
#
# Usage: ./linux_build_mlc.sh [options]
# =============================================================================

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Load central version/dependency configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/versions.sh"

# =============================================================================
# Script Arguments (with defaults)
# =============================================================================
BUILD_VENV="${1:-mlc-llm-venv}"
CUDA="${2:-y}"
CUTLASS="${3:-n}"
CUBLAS="${4:-n}"
ROCM="${5:-n}"
VULKAN="${6:-n}"
OPENCL="${7:-n}"
FLASHINFER="${8:-n}"
CUDA_ARCH="${9:-${CUDA_ARCH_DEFAULT}}"
GITHUB_REPO="${10:-${MLC_LLM_REPO}}"
TVM_SOURCE="${11:-bundled}"  # bundled, relax, or custom
BUILD_WHEELS="${12:-y}"
FORCE_CLONE="${13:-n}"

# =============================================================================
# Variables and Paths
# =============================================================================
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"
MLC_LLM_DIR="${REPO_ROOT}/mlc-llm"
TVM_SOURCE_DIR=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup_on_error() {
    log_error "Build failed! Cleaning up..."
    # Add cleanup logic here if needed
}

trap cleanup_on_error ERR

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed or not in PATH"
        return 1
    fi
    return 0
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

log_info "Running pre-flight checks..."

# Check for conda
if ! check_command conda; then
    log_error "Conda is required but not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Initialize conda for this script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check for git
if ! check_command git; then
    log_error "Git is required but not found."
    exit 1
fi

# Check for CUDA if enabled
if [[ "$CUDA" == "y" ]]; then
    if ! check_command nvcc; then
        log_error "CUDA is enabled but nvcc not found. Please install CUDA toolkit."
        log_info "You can install CUDA using: ./scripts/linux_install_cuda.sh"
        exit 1
    fi
fi

# =============================================================================
# Determine TVM Source Directory
# =============================================================================

case "${TVM_SOURCE}" in
    relax|custom)
        TVM_SOURCE_DIR="${REPO_ROOT}/tvm"
        log_info "Using TVM from: ${TVM_SOURCE_DIR}"
        ;;
    bundled|*)
        TVM_SOURCE_DIR=""
        log_info "Using bundled TVM (from mlc-llm/3rdparty/tvm)"
        ;;
esac

# =============================================================================
# Setup TVM Source (if needed)
# =============================================================================

if [[ "${TVM_SOURCE}" == "relax" ]]; then
    if [[ "${FORCE_CLONE}" == "y" ]] && [[ -d "${TVM_SOURCE_DIR}" ]]; then
        log_info "Force re-clone: removing existing ${TVM_SOURCE_DIR}..."
        rm -rf "${TVM_SOURCE_DIR}"
    fi

    if [[ ! -d "${TVM_SOURCE_DIR}" ]]; then
        log_info "Cloning ${TVM_REPO} ref=${TVM_REF}..."
        git clone "${TVM_REPO}" "${TVM_SOURCE_DIR}"
        git -C "${TVM_SOURCE_DIR}" checkout "${TVM_REF}"
        git -C "${TVM_SOURCE_DIR}" submodule update --init --recursive
    elif [[ "$(git -C "${TVM_SOURCE_DIR}" rev-parse HEAD 2>/dev/null)" != "$(git -C "${TVM_SOURCE_DIR}" rev-parse "${TVM_REF}" 2>/dev/null || echo 'unknown')" ]]; then
        current_tvm_head="$(git -C "${TVM_SOURCE_DIR}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
        log_info "Current TVM HEAD is ${current_tvm_head:0:8}, expected ${TVM_REF}"
        log_info "Switching TVM to ${TVM_REF}..."
        git -C "${TVM_SOURCE_DIR}" remote set-url origin "${TVM_REPO}"
        git -C "${TVM_SOURCE_DIR}" fetch origin
        git -C "${TVM_SOURCE_DIR}" checkout "${TVM_REF}"
        git -C "${TVM_SOURCE_DIR}" submodule update --init --recursive
    else
        log_info "TVM is already at ${TVM_REF}."
    fi

elif [[ "${TVM_SOURCE}" == "custom" ]]; then
    if [[ ! -d "${TVM_SOURCE_DIR}" ]]; then
        log_error "Custom TVM directory not found at ${TVM_SOURCE_DIR}"
        exit 1
    fi
    log_info "Using custom TVM from ${TVM_SOURCE_DIR}"
fi

# =============================================================================
# Clone or Update MLC-LLM
# =============================================================================

if [[ "${FORCE_CLONE}" == "y" ]] && [[ -d "${MLC_LLM_DIR}" ]]; then
    log_info "Force re-clone: removing existing mlc-llm..."
    rm -rf "${MLC_LLM_DIR}"
fi

if [[ ! -d "${MLC_LLM_DIR}" ]]; then
    log_info "Cloning mlc-llm from ${GITHUB_REPO}..."
    git clone --recursive "${GITHUB_REPO}" "${MLC_LLM_DIR}"
    if [[ -n "${MLC_LLM_REF}" ]]; then
        log_info "Checking out MLC_LLM_REF=${MLC_LLM_REF}..."
        git -C "${MLC_LLM_DIR}" checkout "${MLC_LLM_REF}"
        git -C "${MLC_LLM_DIR}" submodule update --init --recursive
    else
        log_info "MLC_LLM_REF is empty; using upstream default branch HEAD."
    fi
else
    if [[ -n "${MLC_LLM_REF}" ]]; then
        current_mlc_head="$(git -C "${MLC_LLM_DIR}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
        expected_mlc_head="$(git -C "${MLC_LLM_DIR}" rev-parse "${MLC_LLM_REF}^{commit}" 2>/dev/null || echo '')"

        if [[ -n "${expected_mlc_head}" && "${current_mlc_head}" == "${expected_mlc_head}" ]]; then
            log_info "mlc-llm already at MLC_LLM_REF=${MLC_LLM_REF} (${current_mlc_head:0:8})"
        else
            log_info "Switching existing mlc-llm checkout to MLC_LLM_REF=${MLC_LLM_REF} (current HEAD: ${current_mlc_head:0:8})"
            git -C "${MLC_LLM_DIR}" remote set-url origin "${GITHUB_REPO}"
            git -C "${MLC_LLM_DIR}" fetch origin
            git -C "${MLC_LLM_DIR}" checkout "${MLC_LLM_REF}"
            git -C "${MLC_LLM_DIR}" submodule update --init --recursive
        fi
    else
        log_info "mlc-llm directory already exists and MLC_LLM_REF is empty; using existing checkout."
    fi
fi

# =============================================================================
# Create Conda Build Environment
# =============================================================================

log_info "Creating/updating build environment: ${BUILD_VENV}"

# Check if environment exists
if conda env list | grep -q "^${BUILD_VENV} "; then
    log_info "Environment ${BUILD_VENV} already exists, updating..."
else
    log_info "Creating new environment ${BUILD_VENV}..."
    conda create -n "${BUILD_VENV}" -c "${CONDA_CHANNEL}" --yes \
        "cmake>=${CMAKE_MIN_VERSION}" \
        rust \
        git \
        "${PYTHON_PACKAGE_SPEC}" \
        "${PYTHON_ABI_SPEC}" \
        pip \
        git-lfs
fi

log_success "Build environment ready: ${BUILD_VENV}"

# =============================================================================
# Activate Build Environment
# =============================================================================

conda activate "${BUILD_VENV}"
CONDA_PYTHON="${CONDA_PREFIX}/bin/python"

# Get number of CPU cores for parallel build
NCORES=$(nproc)
log_info "Building with ${NCORES} parallel jobs"

# =============================================================================
# Configure and Build MLC-LLM
# =============================================================================

mkdir -p "${MLC_LLM_DIR}/build"
cd "${MLC_LLM_DIR}/build"

# Generate CMake config
log_info "Generating CMake configuration..."
printf "%s\n%s\n%s\n%s\n%s\n%s\nn\n%s\n%s\n\n\n" \
    "${TVM_SOURCE_DIR}" \
    "${CUDA}" \
    "${CUTLASS}" \
    "${CUBLAS}" \
    "${ROCM}" \
    "${VULKAN}" \
    "${OPENCL}" \
    "${FLASHINFER}" | python3 ../cmake/gen_cmake_config.py

# Configure CUDA-specific settings
if [[ "$CUDA" == "y" ]]; then
    # Inject CUDA_ARCHITECTURES into config.cmake
    echo "set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH} CACHE STRING \"CUDA architectures\" FORCE)" >> config.cmake
    
    # Disable Thrust due to known compilation failures
    sed -i 's/set(USE_THRUST.*/set(USE_THRUST OFF)/' config.cmake
    
    log_info "Configured for CUDA architecture: ${CUDA_ARCH}"
fi

# Run CMake
if [[ "$CUDA" == "y" ]]; then
    log_info "Configuring with CUDA support..."
    
    # Find nvcc
    if command -v nvcc >/dev/null 2>&1; then
        NVCC_PATH="$(command -v nvcc)"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        NVCC_PATH="/usr/local/cuda/bin/nvcc"
    elif [[ -x /usr/bin/nvcc ]]; then
        NVCC_PATH="/usr/bin/nvcc"
    else
        log_error "nvcc not found. Please install CUDA toolkit or add it to PATH."
        exit 1
    fi

    # Resolve symlinks
    NVCC_REAL="$(readlink -f "${NVCC_PATH}" 2>/dev/null || echo "${NVCC_PATH}")"
    CUDA_BIN_DIR="$(dirname "${NVCC_REAL}")"
    CUDA_HOME="$(dirname "${CUDA_BIN_DIR}")"

    export PATH="${CUDA_BIN_DIR}:${PATH}"
    export CUDACXX="${NVCC_REAL}"
    export CUDA_HOME="${CUDA_HOME}"

    if [[ -d "${CUDA_HOME}/lib64" ]]; then
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    fi

    log_info "Using CUDA from: ${CUDA_HOME}"
    log_info "CUDA compiler: ${NVCC_REAL}"
    
    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DCMAKE_CUDA_COMPILER="${NVCC_REAL}"
else
    log_info "Building with CPU-only support..."
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
fi

# Build
log_info "Building MLC-LLM (this may take a while)..."
cmake --build . --parallel "${NCORES}"

log_success "MLC-LLM build completed!"

# =============================================================================
# Build Python Wheels (optional)
# =============================================================================

if [[ "${BUILD_WHEELS}" == "y" ]]; then
    log_info "Building Python wheels..."
    mkdir -p "${WHEELS_DIR}"

    cd "${MLC_LLM_DIR}/python"
    "${CONDA_PYTHON}" -m pip install --quiet build
    "${CONDA_PYTHON}" -m build --wheel --outdir "${WHEELS_DIR}"

    log_success "MLC-LLM wheel created in ${WHEELS_DIR}"
else
    log_info "Skipping wheel build (BUILD_WHEELS=${BUILD_WHEELS})"
fi

# =============================================================================
# Cleanup and Exit
# =============================================================================

conda deactivate
log_success "MLC-LLM build completed successfully!"
log_info "Next steps:"
log_info "  1. Run './mlc-cli install tvm' to install TVM wheel"
log_info "  2. Run './mlc-cli install mlc' to install MLC wheel"
log_info "  3. Run your model with './mlc-cli run'"
