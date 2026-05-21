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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
# Pre-flight Checks
# =============================================================================

if ! command -v conda &> /dev/null; then
    log_error "Conda is not installed"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${CLI_VENV} "; then
    log_error "CLI environment '${CLI_VENV}' not found. Please run build first."
fi

# =============================================================================
# Model Setup
# =============================================================================

mkdir -p "${MODELS_DIR}"

# Clone model if URL provided
if [[ -n "${MODEL_URL}" ]]; then
    if [[ -z "${MODEL_NAME}" ]]; then
        MODEL_NAME="$(basename "${MODEL_URL}")"
    fi
    MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}"
    if [[ ! -d "${MODEL_PATH}" ]]; then
        log_info "Cloning model from ${MODEL_URL}..."
        git clone --depth 1 "${MODEL_URL}" "${MODEL_PATH}"
    else
        log_info "Model already exists at ${MODEL_PATH}"
    fi
else
    if [[ -n "${MODEL_NAME}" ]]; then
        MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}"
    else
        log_error "Model name is required"
    fi
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
    "chat"
    "${MODEL_PATH}"
    "--device" "${DEVICE}"
)

# Add overrides if provided
if [[ -n "${OVERRIDES}" ]]; then
    RUN_ARGS+=("--overrides" "${OVERRIDES}")
fi

# Add model lib if provided
if [[ -n "${MODEL_LIB}" ]]; then
    RUN_ARGS+=("--model-lib" "${MODEL_LIB}")
fi

# Execute
python -m mlc_llm "${RUN_ARGS[@]}"

conda deactivate
log_success "Model run completed!"
