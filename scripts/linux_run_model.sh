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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)
MODELS_DIR="${REPO_ROOT}/models"

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

if ! conda env list | grep -q "^${CLI_VENV})"; then
    log_error "CLI environment '${CLI_VENV}' not found. Please run build first."
fi

# =============================================================================
# Model Setup
# =============================================================================

mkdir -p "${MODELS_DIR}"

# Clone model if URL provided
if [[ -n "${MODEL_URL}" ]]; then
    cd "${MODELS_DIR}"
    log_info "Cloning model from ${MODEL_URL}..."
    git clone --depth 1 "${MODEL_URL}" "$(basename "${MODEL_URL}")"
    cd ..
fi

# Determine model path
MODEL_PATH=""
if [[ -n "${MODEL_NAME}" ]]; then
    MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}"
else
    log_error "Model name is required"
fi

# =============================================================================
# Run Model
# =============================================================================

conda activate "${CLI_VENV}"

log_info "Running model with MLC-LLM..."
log_info "Model: ${MODEL_NAME}"
log_info "Device: ${DEVICE}"

# Build run command
RUN_ARGS=(
    "${CLI_VENV}" \
    "${MODEL_PATH}" \
    "${DEVICE}"
)

# Add overrides if provided
if [[ -n "${OVERRIDES}" ]]; then
    RUN_ARGS="${RUN_ARGS} --overrides ${OVERRIDES}"
fi

# Add model lib if provided
if [[ -n "${MODEL_LIB}" ]]; then
    RUN_ARGS="${RUN_ARGS} --model-lib ${MODEL_LIB}"
fi

# Execute
python -m mlc_llm.cli run ${RUN_ARGS}

popd
conda deactivate
log_success "Model run completed!"
