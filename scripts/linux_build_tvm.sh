#!/usr/bin/env bash
set -eu

# =============================================================================
# TVM Build Script for Linux
# =============================================================================

BUILD_VENV="${1:-tvm-build-venv}"
TVM_SOURCE="${2:-bundled}"
BUILD_WHEELS="${3:-y}"
FORCE_CLONE="${4:-n}"
CUDA_ARCH="${5:-86}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"
TVM_DIR="${REPO_ROOT}/tvm"

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
    rm -rf "${TVM_DIR}"
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

if [[ "$TVM_SOURCE" == "relax" ]] || [[ "$TVM_SOURCE" == "custom" ]]; then
    if [[ "$FORCE_CLONE" == "y" ]] && [ -d "$TVM_DIR" ]; then
        log_info "Force re-clone: removing existing TVM directory..."
        rm -rf "${TVM_DIR}"
    fi
    if [ ! -d "$TVM_DIR" ]; then
        if [[ "$TVM_SOURCE" == "relax" ]]; then
            log_info "Cloning mlc-ai/relax on mlc branch..."
            git clone --recursive -b mlc https://github.com/mlc-ai/relax.git "${TVM_DIR}"
        fi
    else
        log_info "Using TVM from ${TVM_DIR}"
    fi
else
    log_info "Will use bundled TVM (mlc-llm builds this internally)"
    log_info "This script is typically called by linux_build_mlc.sh"
    exit 0
fi

# =============================================================================
# Conda Environment
# =============================================================================

if conda env list | grep -q "^${BUILD_VENV})" &> /dev/null; then
    log_info "Creating conda environment: ${BUILD_VENV}"
    conda create -y -n "${BUILD_VENV}" -c conda-forge \
        "cmake>=3.24" \
        rust \
        git \
        python=3.11 \
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
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
make -j"$(nproc)"
log_success "TVM build completed!"

# =============================================================================
# Build Python Wheel (optional)
# =============================================================================

if [[ "${BUILD_WHEELS}" == "y" ]]; then
    log_info "Building TVM Python wheel..."
    mkdir -p "${WHEELS_DIR}"

    cd "${TVM_DIR}"/python
    python -m pip install --quiet build
    python -m build --wheel --outdir "${WHEELS_DIR}"
    cd ../build

    log_success "TVM wheel created in ${WHEELS_DIR}"
else
    log_info "Skipping TVM wheel build"
fi

popd
conda deactivate
log_success "TVM build completed successfully!"
