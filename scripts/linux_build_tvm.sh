#!/usr/bin/env bash
set -eu

# =============================================================================
# TVM Build Script for Linux
# =============================================================================

# Load central version/dependency configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/versions.sh"

BUILD_VENV="${1:-tvm-build-venv}"
TVM_SOURCE="${2:-bundled}"
BUILD_WHEELS="${3:-y}"
FORCE_CLONE="${4:-n}"
CUDA_ARCH="${5:-${CUDA_ARCH_DEFAULT}}"

# SCRIPT_DIR already set above when sourcing versions.sh
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"
TVM_DIR="${REPO_ROOT}/tvm"
MLC_LLM_DIR="${REPO_ROOT}/mlc-llm"

RED='\033[1;31m'
GREEN='\033[0;32m'
BLUE='\033[1;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

cleanup_on_error() {
    log_error "TVM build failed! Cleaning up..."
    rm -rf "${TVM_DIR}/build" 2>/dev/null || true

    if [[ "${TVM_SOURCE:-}" != "bundled" ]]; then
        rm -rf "${TVM_DIR}" 2>/dev/null || true
    fi
}

trap cleanup_on_error ERR

# Check for conda
if ! command -v conda &> /dev/null; then
    log_error "Conda is not installed. Please install conda first."
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# =============================================================================
# TVM Source Setup
# =============================================================================

if [[ "$TVM_SOURCE" == "bundled" ]]; then
    if [[ "$FORCE_CLONE" == "y" ]] && [ -d "$MLC_LLM_DIR" ]; then
        log_info "Force re-clone: removing existing MLC-LLM directory..."
        rm -rf "${MLC_LLM_DIR}"
    fi

    if [ ! -d "$MLC_LLM_DIR" ]; then
        log_info "Cloning MLC-LLM from ${MLC_LLM_REPO}..."
        git clone --recursive "${MLC_LLM_REPO}" "${MLC_LLM_DIR}"
        if [[ -n "${MLC_LLM_REF}" ]]; then
            log_info "Checking out MLC_LLM_REF=${MLC_LLM_REF}..."
            git -C "${MLC_LLM_DIR}" checkout "${MLC_LLM_REF}"
            git -C "${MLC_LLM_DIR}" submodule update --init --recursive
        fi
    fi

    TVM_DIR="${MLC_LLM_DIR}/3rdparty/tvm"
    if [ ! -d "$TVM_DIR" ]; then
        log_error "Bundled TVM directory not found at ${TVM_DIR}"
        exit 1
    fi

    log_info "Using bundled TVM from ${TVM_DIR}"

elif [[ "$TVM_SOURCE" == "relax" ]] || [[ "$TVM_SOURCE" == "custom" ]]; then
    if [[ "$FORCE_CLONE" == "y" ]] && [ -d "$TVM_DIR" ]; then
        log_info "Force re-clone: removing existing TVM directory..."
        rm -rf "${TVM_DIR}"
    fi
    if [ ! -d "$TVM_DIR" ]; then
        if [[ "$TVM_SOURCE" == "relax" ]]; then
            log_info "Cloning ${TVM_REPO} ref=${TVM_REF}..."
            git clone "${TVM_REPO}" "${TVM_DIR}"
            git -C "${TVM_DIR}" checkout "${TVM_REF}"
            git -C "${TVM_DIR}" submodule update --init --recursive
        fi
    elif [[ "$(git -C "${TVM_DIR}" rev-parse HEAD 2>/dev/null)" != "$(git -C "${TVM_DIR}" rev-parse "${TVM_REF}" 2>/dev/null || echo 'unknown')" ]]; then
        current_tvm_head="$(git -C "${TVM_DIR}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
        log_info "Current TVM HEAD is ${current_tvm_head:0:8}, expected ${TVM_REF}"
        log_info "Switching TVM to ${TVM_REF}..."
        git -C "${TVM_DIR}" remote set-url origin "${TVM_REPO}"
        git -C "${TVM_DIR}" fetch origin
        git -C "${TVM_DIR}" checkout "${TVM_REF}"
        git -C "${TVM_DIR}" submodule update --init --recursive
    else
        log_info "TVM is already at ${TVM_REF}."
    fi
else
    log_error "Unsupported TVM_SOURCE=${TVM_SOURCE}"
    exit 1
fi

# =============================================================================
# Conda Environment
# =============================================================================

if ! conda env list | grep -q "^${BUILD_VENV} " &> /dev/null; then
    log_info "Creating conda environment: ${BUILD_VENV}"
    conda create -y -n "${BUILD_VENV}" -c "${CONDA_CHANNEL}" \
        "cmake>=${CMAKE_MIN_VERSION}" \
        ninja \
        rust \
        git \
        "${PYTHON_PACKAGE_SPEC}" \
        "${PYTHON_ABI_SPEC}" \
        pip
else
    log_info "Environment '${BUILD_VENV}' already exists, using it"
fi

conda activate "${BUILD_VENV}"

# =============================================================================
# Build TVM
# =============================================================================

cd "${TVM_DIR}" || exit 1
mkdir -p build
cd build

log_info "Configuring TVM build..."

cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
cmake --build . --parallel "$(nproc)"
log_success "TVM build completed!"

# =============================================================================
# Build Python Wheel (optional)
# =============================================================================

if [[ "${BUILD_WHEELS}" == "y" ]]; then
    log_info "Building TVM Python wheel..."
    mkdir -p "${WHEELS_DIR}"

    cd "${TVM_DIR}"
    python -m pip install --quiet build
    python -m build --wheel --outdir "${WHEELS_DIR}"

    log_success "TVM wheel created in ${WHEELS_DIR}"

    TVM_FFI_DIR="${TVM_DIR}/3rdparty/tvm-ffi"
    if [[ -d "${TVM_FFI_DIR}" ]]; then
        log_info "Building bundled apache-tvm-ffi wheel..."
        cd "${TVM_FFI_DIR}"
        python -m build --wheel --outdir "${WHEELS_DIR}"
        log_success "apache-tvm-ffi wheel created in ${WHEELS_DIR}"
    else
        log_info "No bundled tvm-ffi source found; skipping apache-tvm-ffi wheel build"
    fi
else
    log_info "Skipping TVM wheel build"
fi

conda deactivate
log_success "TVM build completed successfully!"
