#!/usr/bin/env bash
set -eu

# =============================================================================
# Configuration
# =============================================================================
CLI_VENV="${1:-mlc-cli-venv}"
TVM_WHEEL="${2:-}"
MLC_WHEEL="${3:-}"
INSTALL_MODE="${4:-source}"  # source or wheel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)
WHEELS_DIR="${REPO_ROOT}/wheels"

RED='\033[1;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Pre-flight Checks
# =============================================================================

if ! command -v conda &> /dev/null; then
    log_error "Conda is required but not installed"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# =============================================================================
# Find Wheels
# =============================================================================

find_wheel() {
    local pattern="$1"
    local wheels=($(ls "${WHEELS_DIR}"/${pattern}"*.whl 2>/dev/null))
    if [ ${#wheels[@]} -gt 0 ]; then
        echo "${wheels[0]}"
        return 0
    fi
    log_error "No wheel found matching pattern: ${pattern}"
    return 1
}

# =============================================================================
# Main
# =============================================================================

log_info "Installing MLC-LLM into CLI environment..."

conda activate "${CLI_VENV}" || {
    log_error "Failed to activate environment: ${CLI_VENV}"
    exit 1
}

# Install TVM first if in source mode
if [[ "${INSTALL_MODE}" == "source" ]]; then
    log_info "Installing TVM wheel first..."
    TVM_WHEEL_PATH=$(find_wheel "tvm")
    pip install --force "${TVM_WHEEL_PATH}"
    log_success "TVM wheel installed"
fi

# Install MLC wheel
log_info "Installing MLC wheel..."
MLC_WHEEL_PATH=$(find_wheel "mlc")
pip install --force "${MLC_WHEEL_PATH}"
log_success "MLC wheel installed"

popd
conda deactivate

log_success "Installation completed successfully!"
log_info ""
log_info "You can now use the CLI environment '${CLI_VENV}' to run models."
log_info "  conda activate ${CLI_VENV}"
log_info "  python -m mlc_llm.cli --help"
