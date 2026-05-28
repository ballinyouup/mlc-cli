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
        if [[ -d "${MODELS_DIR}/${MODEL_NAME}" ]]; then
            MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}"
        elif [[ -d "${REPO_ROOT}/dist/${MODEL_NAME}" ]]; then
            log_info "Model not found in models/, using dist/${MODEL_NAME}"
            MODEL_PATH="${REPO_ROOT}/dist/${MODEL_NAME}"
        else
            log_error "Model '${MODEL_NAME}' not found in models/ or dist/. Run quantize first or pass --model-url."
        fi
    else
        log_error "Model name is required"
    fi
fi

# =============================================================================
# Run Model
# =============================================================================

CONDA_BASE="$(conda info --base)"
CONDA_BIN="${CONDA_BASE}/bin/conda"

log_info "Running model with MLC-LLM..."
log_info "Model: ${MODEL_NAME}"
log_info "Device: ${DEVICE}"

# Build run command
# conda activate silently fails in non-interactive subshells (CONDA_PREFIX stays
# empty, python resolves to /usr/bin/python).  Use `conda run` instead, which
# correctly invokes the env's own Python without requiring an interactive shell.
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

# Export required environment variables for TVM and MLC
export TVM_HOME="${REPO_ROOT}/tvm"
export PYTHONPATH="${REPO_ROOT}/tvm/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO_ROOT}/tvm/build/lib:${REPO_ROOT}/mlc-llm/build/lib:${LD_LIBRARY_PATH:-}"

# Execute via conda run so stdin/stdout stream correctly (--no-capture-output)
"${CONDA_BIN}" run --no-capture-output -n "${CLI_VENV}" \
    python -m mlc_llm "${RUN_ARGS[@]}"

log_success "Model run completed!"
