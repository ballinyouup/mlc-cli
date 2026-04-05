#!/usr/bin/env bash
set -eu

# =============================================================================
# Install Pre-built Wheels Script for macOS
# =============================================================================

CLI_VENV="${1:-mlc-cli-venv}"
WHEELS_DIR="${2:-wheels}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/${WHEELS_DIR}"

RED='\033[1;31m'
GREEN='\033[0;32m'
BLUE='\033[1;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
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

# Count wheels
TVM_WHEELS=($(ls "${WHEELS_DIR}"/tvm*.whl 2>/dev/null || true))
MLC_WHEELS=($(ls "${WHEELS_DIR}"/mlc*.whl 2>/dev/null || true))

if [ ${#TVM_WHEELS[@]} -eq 0 ] && [ ${#MLC_WHEELS[@]} -eq 0 ]; then
    log_error "No wheels found in ${WHEELS_DIR}"
fi

log_info "Found ${#TVM_WHEELS[@]} TVM wheels and ${#MLC_WHEELS[@]} MLC wheels"

# =============================================================================
# Environment Setup
# =============================================================================

if ! conda env list | grep -q "^${CLI_VENV})"; then
    log_info "Creating environment: ${CLI_VENV}"
    conda create -y -n "${CLI_VENV}" -c conda-forge \
        "cmake>=3.24" \
        python=3.11 \
        pip
else
    log_info "Using existing environment: ${CLI_VENV}"
fi

conda activate "${CLI_VENV}"

# =============================================================================
# Install Wheels
# =============================================================================

# Install TVM wheel first
if [ ${#TVM_WHEELS[@]} -gt 0 ]; then
    log_info "Installing TVM wheel..."
    pip install --force "${TVM_WHEELS[0]}"
    log_success "TVM wheel installed"
fi

# Install MLC wheel
if [ ${#MLC_WHEELS[@]} -gt 0 ]; then
    log_info "Installing MLC wheel..."
    pip install --force "${MLC_WHEELS[0]}"
    log_success "MLC wheel installed"
fi

# =============================================================================
# Verify Installation
# =============================================================================

log_info "Verifying installation..."

python -c "import tvm; print(f'TVM version: {tvm.__version__}')" || true
python -c "import mlc_llm; print('MLC-LLM imported')" || true

popd
conda deactivate
log_success "Wheel installation completed!"
log_info ""
log_info "To use the CLI:"
log_info "  conda activate ${CLI_VENV}"
log_info "  python -m mlc_llm.cli --help"
