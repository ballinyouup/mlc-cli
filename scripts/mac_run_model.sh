#!/usr/bin/env bash
set -eu

# =============================================================================
# Configuration
# =============================================================================
CLI_VENV="${1:-mlc-cli-venv}"
MODEL_URL="${2:-}"
MODEL_NAME="${3:-}"
DEVICE="${4:-metal}"
OVERRIDES="${5:-}"
MODEL_LIB="${6:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"

RED='\033[1;31m'
GREEN='\033[0;32m'
BLUE='\033[1;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# Check Conda
# =============================================================================

if ! command -v conda &> /dev/null; then
    log_error "Conda is not installed"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# =============================================================================
# Setup Model
# =============================================================================

mkdir -p "${MODELS_DIR}"

if [[ -n "${MODEL_URL}" ]]; then
    log_info "Cloning model from ${MODEL_URL}..."
    cd "${MODELS_DIR}"
    git clone --depth 1 "${MODEL_URL}" "$(basename "${MODEL_URL}")"
    cd ..
fi

# Determine model path
if [[ -n "${MODEL_NAME}" ]]; then
    MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}"
else
    log_error "Model name is required"
fi

# =============================================================================
# Activate Environment and Run
# =============================================================================

conda activate "${CLI_VENV}"

log_info "Running model: ${MODEL_NAME} on ${DEVICE}"

# Build MLC CLI command
MLC_ARGS=(
    "chat" \
    "${MODEL_PATH}" \
    "--device" "${DEVICE}"
)

if [[ -n "${OVERRIDES}" ]]; then
    MLC_ARGS="${MLC_ARGS} --overrides ${OVERRIDES}"
fi

if [[ -n "${MODEL_LIB}" ]]; then
    MLC_ARGS="${MLC_ARGS} --model-lib-path ${MODEL_LIB}"
fi

log_info "Running: mlc_llm ${MLC_ARGS}"
python -m mlc_llm ${MLC_ARGS}

popd
conda deactivate
log_success "Model run completed!"
