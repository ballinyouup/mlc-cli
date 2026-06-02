#!/usr/bin/env bash
set -eu

# =============================================================================
# Install Pre-built Wheels Script for Linux
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/versions.sh"
source "${SCRIPT_DIR}/lib/wheel_selection.sh"

CLI_VENV="${1:-mlc-cli-venv}"
WHEELS_DIR="${2:-wheels}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/${WHEELS_DIR}"

RED='\033[1;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# Pre-flight Checks
# =============================================================================

if ! command -v conda &> /dev/null; then
    log_error "Conda is not installed"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# Check for wheels directory
if [ ! -d "${WHEELS_DIR}" ]; then
    log_error "Wheels directory not found: ${WHEELS_DIR}"
fi

# Count TVM wheels
TVM_WHEELS=($(ls "${WHEELS_DIR}"/tvm-*-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-*.whl 2>/dev/null || true))
if [ ${#TVM_WHEELS[@]} -gt 1 ]; then
    log_warning "Multiple TVM wheels found. Using the first one: $(basename "${TVM_WHEELS[0]}")"
fi

if ! find_mlc_wheel; then
    log_error "Failed to select MLC wheel. Please check the errors above."
fi

if [ ${#TVM_WHEELS[@]} -eq 0 ] && [ -z "${MLC_WHEEL_PATH}" ]; then
    log_error "No wheels found in ${WHEELS_DIR} matching PYTHON_CP_TAG=${PYTHON_CP_TAG}"
fi

if [ -n "${MLC_WHEEL_PATH}" ]; then
    log_info "Found ${#TVM_WHEELS[@]} TVM wheels and 1 MLC wheel"
else
    log_info "Found ${#TVM_WHEELS[@]} TVM wheels and 0 MLC wheels"
fi

# =============================================================================
# Environment Setup
# =============================================================================

if ! conda env list | grep -q "^${CLI_VENV} "; then
    log_info "Creating environment: ${CLI_VENV}"
    conda create -y -n "${CLI_VENV}" -c "${CONDA_CHANNEL}" "${PYTHON_PACKAGE_SPEC}" "${PYTHON_ABI_SPEC}" pip pytest psutil
else
    log_info "Using existing environment: ${CLI_VENV}"
    conda install -y -n "${CLI_VENV}" -c "${CONDA_CHANNEL}" pytest psutil
fi


# =============================================================================
# Install Wheels
# =============================================================================

# Install TVM wheel first (MLC depends on it)
if [ ${#TVM_WHEELS[@]} -gt 0 ]; then
    log_info "Installing TVM wheel..."
    conda run --no-capture-output -n "${CLI_VENV}" python -m pip install --force-reinstall "${TVM_WHEELS[0]}"
    log_success "TVM wheel installed"
fi

# Install MLC wheel
if [ -n "${MLC_WHEEL_PATH}" ]; then
    log_info "Installing MLC wheel..."
    conda run --no-capture-output -n "${CLI_VENV}" python -m pip install --force-reinstall "${MLC_WHEEL_PATH}"
    log_success "MLC wheel installed"
fi

# =============================================================================
# Verify Installation
# =============================================================================

log_info "Verifying installation..."

conda run --no-capture-output -n "${CLI_VENV}" python -c "from tvm import register_global_func; print('TVM import OK')" || log_warning "TVM import check failed"
conda run --no-capture-output -n "${CLI_VENV}" python -c "import mlc_llm; print('MLC-LLM import OK')" || log_warning "MLC-LLM import check failed"

log_success "Wheel installation completed!"
log_info ""
log_info "To use the CLI:"
log_info "  conda activate ${CLI_VENV}"
log_info "  python -m mlc_llm --help"
