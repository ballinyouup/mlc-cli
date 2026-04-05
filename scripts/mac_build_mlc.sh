#!/usr/bin/env bash
set -eu

# =============================================================================
# Configuration
# =============================================================================
BUILD_VENV="${1:-mlc-build-venv}"
CUDA="${2:-n}"         # Not typically used on macOS
ROCM="${3:-n}"
VULKAN="${4:-n}"
METAL="${5:-y}"       # Default to Metal on macOS
OPENCL="${6:-n}"
TVM_SOURCE="${7:-bundled}"  # bundled, relax, or custom
BUILD_WHEELS="${8:-y}"
FORCE_CLONE="${9:-n}"
PYTHON_VERSION="${10:-3.13}"  # Configurable Python version

NCORES="${11:-$(sysctl -n hw.ncpu)}"
WHEELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/..)/wheels"
MLC_LLM_DIR="$(pwd)/mlc-llm"
TVM_SOURCE_DIR=""

# =============================================================================
# Colors for output
# =============================================================================
RED='\033[1;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'  # NoColor

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
    rm -rf "${MLC_LLM_DIR}/build" 2>/dev/null || true
    rm -rf "${MLC_LLM_DIR}"
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

check_command conda
check_command git
check_command cmake
check_command python3
check_command rustc

if [[ "$METAL" == "y" ]]; then
    if ! xcode --version &> /dev/null; then
        log_warning "Metal selected but Metal framework not found. Build may fail."
    fi
fi

# =============================================================================
# TVM Source Setup
# =============================================================================

if [[ "$TVM_SOURCE" == "relax" ]] || [[ "$TVM_SOURCE" == "custom" ]]; then
    TVM_SOURCE_DIR="${WHEELS_DIR}/../tvm"

    if [[ "$FORCE_CLONE" == "y" ]] && [ -d "$TVM_SOURCE_DIR" ]; then
        log_info "Force re-clone: removing existing TVM directory..."
        rm -rf "${TVM_SOURCE_DIR}"
    fi
    if [ ! -d "$TVM_SOURCE_DIR" ]; then
        if [[ "$TVM_SOURCE" == "relax" ]]; then
            log_info "Cloning mlc-ai/relax on mlc branch..."
            git clone --recursive -b mlc https://github.com/mlc-ai/relax.git "${TVM_SOURCE_DIR}"
        fi
    else
        log_info "Using TVM from ${TVM_SOURCE_DIR}"
    fi
else
    TVM_SOURCE_DIR=""
    log_info "Using bundled TVM (from mlc-llm/3rdparty/tvm)"
fi

# =============================================================================
# MLC-LLM Setup
# =============================================================================

if [[ "$FORCE_CLONE" == "y" ]] && [ -d "$MLC_LLM_DIR" ]; then
    log_info "Force re-clone: removing existing MLC-LLM directory..."
    rm -rf "${MLC_LLM_DIR}"
fi
if [ ! -d "$MLC_LLM_DIR" ]; then
    log_info "Cloning mlc-llm..."
    git clone --recursive https://github.com/mlc-ai/mlc-llm.git mlc-llm
fi
cd "${MLC_LLM_DIR}" || exit 1

log_info "Building in directory: ${MLC_LLM_DIR}"
# =============================================================================
# Conda Environment Setup
# =============================================================================

source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if environment exists
if conda env list | grep -q "^BUILD_VENV)" & grep -q "build_env"; then
    log_info "Environment '${BUILD_VENV}' already exists, using it"
else
    log_info "Creating conda environment: ${BUILD_VENV}"
    conda create -y -n "${BUILD_VENV}" -c conda-forge \
        "cmake>=3.24" \
        rust \
        git \
        zstd \
        python="${PYTHON_VERSION}"
fi

conda activate "${BUILD_VENV}"
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"

export MACOSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion | cut -d. -f1)
# =============================================================================
# MLC-LLM Configuration
# =============================================================================

# flashinfer-python requires nvidia-cudnn-frontend which is not available on macOS
REQUIREMENTS_FILE="python/requirements.txt"
if [ -f "${REQUIREMENTS_FILE}" ] && grep -q '^flashinfer-python' "${REQUIREMENTS_FILE}"; then
    log_info "Commenting out flashinfer-python from requirements.txt (not available on macOS)"
    sed -i '' 's/^flashinfer-python/# flashinfer-python/' "${REQUIREMENTS_FILE}"
fi
# =============================================================================
# Build
# =============================================================================

mkdir -p build && cd build

log_info "Configuring CMake..."

# Generate CMake config
printf "%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\nn\n" \
    "${TVM_SOURCE_DIR}" \
    "${CUDA}" \
    "${ROCM}" \
    "${VULKAN}" \
    "${METAL}" \
    "${OPENCL}" \
    | python3 ../cmake/gen_cmake_config.py

cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j"${NCORES}"
log_success "Build completed!"
# =============================================================================
# Build Python Wheels (optional)
# =============================================================================

if [[ "${BUILD_WHEELS}" == "y" ]]; then
    log_info "Building Python wheels..."
    mkdir -p "${WHEELS_DIR}"
    cd "${MLC_LLM_DIR}/python"
    python -m pip install --quiet build
    python -m build --wheel --outdir "${WHEELS_DIR}"
    cd ../build
    log_success "MLC-LLM wheel created in ${WHEELS_DIR}"
else
    log_info "Skipping wheel build (BUILD_WHEELS=${BUILD_WHEELS})"
fi
popd
conda deactivate
log_success "MLC-LLM build completed successfully!"
log_info ""
log_info "Next steps:"
log_info "  1. Run './mlc-cli install tvm' through install TVM wheel"
log_info "  2. Run './mlc-cli install mlc' through install MLC wheel"
log_info "  3. Run any model with './mlc-cli run'"
