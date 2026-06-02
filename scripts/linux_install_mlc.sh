#!/usr/bin/env bash
set -eu

# =============================================================================
# Configuration
# =============================================================================
CLI_VENV="${1:-mlc-cli-venv}"
TVM_SOURCE="${2:-bundled}"
INSTALL_MODE="${3:-source}"  # source or wheel

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
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# Pre-flight Checks
# =============================================================================

if ! command -v conda &> /dev/null; then
    log_error "Conda is required but not installed"
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
}

# Install TVM first if in source mode and a standalone TVM wheel exists.
# In bundled mode linux_build_mlc.sh does not produce a tvm-*.whl; TVM is
# embedded inside the mlc_llm wheel, so this step is skipped there.
if [[ "${INSTALL_MODE}" == "source" ]]; then
    if [[ "${TVM_SOURCE}" == "bundled" ]]; then
        log_info "Bundled TVM mode — skipping standalone TVM wheel install"
    else
        TVM_WHEELS=($(ls "${WHEELS_DIR}"/tvm-*-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-*.whl 2>/dev/null || true))

        if [[ ${#TVM_WHEELS[@]} -eq 0 ]]; then
            log_error "No ABI-matching standalone TVM wheel found in ${WHEELS_DIR} for TVM_SOURCE=${TVM_SOURCE}. Run build first or use TVM_SOURCE=bundled."
        fi

        if [[ ${#TVM_WHEELS[@]} -gt 1 ]]; then
            printf '%s\n' "${TVM_WHEELS[@]}"
            log_error "Multiple ABI-matching TVM wheels found for TVM_SOURCE=${TVM_SOURCE}. Remove stale wheels and retry."
        fi

        log_info "Installing standalone TVM wheel for TVM_SOURCE=${TVM_SOURCE}..."
        python -m pip install --force-reinstall "${TVM_WHEELS[0]}"
        log_success "TVM wheel installed"
    fi
fi

# Install MLC wheel
log_info "Installing MLC wheel..."
if ! find_mlc_wheel; then
    log_error "Failed to select MLC wheel. Please check the errors above."
fi
if [[ -z "${MLC_WHEEL_PATH}" ]]; then
    log_error "No ABI-matching MLC wheel found in ${WHEELS_DIR}. Run build first."
fi
python -m pip install --force-reinstall "${MLC_WHEEL_PATH}"
log_success "MLC wheel installed"

conda deactivate

log_success "Installation completed successfully!"
log_info ""
log_info "You can now use the CLI environment '${CLI_VENV}' to run models."
log_info "  conda activate ${CLI_VENV}"
log_info "  python -m mlc_llm --help"
