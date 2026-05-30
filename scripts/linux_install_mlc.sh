#!/usr/bin/env bash
set -eu

# =============================================================================
# Configuration
# =============================================================================
CLI_VENV="${1:-mlc-cli-venv}"
TVM_WHEEL="${2:-}"
MLC_WHEEL="${3:-}"
INSTALL_MODE="${4:-source}"  # source or wheel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/versions.sh"
source "${SCRIPT_DIR}/lib/wheel_selection.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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


# =============================================================================
# Main
# =============================================================================

log_info "Installing MLC-LLM into CLI environment..."

conda activate "${CLI_VENV}" || {
    log_error "Failed to activate environment: ${CLI_VENV}"
    exit 1
}

# Install TVM first if in source mode and a standalone TVM wheel exists.
# In bundled mode linux_build_mlc.sh does not produce a tvm-*.whl; TVM is
# embedded inside the mlc_llm wheel, so this step is skipped there.
if [[ "${INSTALL_MODE}" == "source" ]]; then
    if [[ -n "${TVM_WHEEL}" ]]; then
        log_info "Using explicit TVM_WHEEL argument: ${TVM_WHEEL}"
        if [[ ! -f "${TVM_WHEEL}" ]]; then
            log_error "Explicit TVM_WHEEL path does not exist: ${TVM_WHEEL}"
            exit 1
        fi
        TVM_WHEELS=("${TVM_WHEEL}")
    else
        TVM_WHEELS=($(ls "${WHEELS_DIR}"/tvm-*-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-*.whl 2>/dev/null || true))
        if [ ${#TVM_WHEELS[@]} -gt 1 ]; then
            log_warning "Multiple TVM wheels found. Using the first one: $(basename "${TVM_WHEELS[0]}")"
        fi
    fi
    if [[ ${#TVM_WHEELS[@]} -gt 0 ]] && [[ -f "${TVM_WHEELS[0]}" ]]; then
        log_info "Installing TVM wheel first..."
        python -m pip install --force "${TVM_WHEELS[0]}"
        log_success "TVM wheel installed"
    else
        log_info "No standalone TVM wheel found in ${WHEELS_DIR} matching tvm-*-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-*.whl (bundled mode — skipping TVM wheel install)"
    fi
fi

# Install MLC wheel
log_info "Installing MLC wheel..."
find_mlc_wheel
if [[ -n "${MLC_WHEEL_PATH}" ]]; then
    python -m pip install --force "${MLC_WHEEL_PATH}"
    log_success "MLC wheel installed"
fi

conda deactivate

log_success "Installation completed successfully!"
log_info ""
log_info "You can now use the CLI environment '${CLI_VENV}' to run models."
log_info "  conda activate ${CLI_VENV}"
log_info "  python -m mlc_llm --help"
