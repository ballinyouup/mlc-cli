#!/usr/bin/env bash
set -eu

# =============================================================================
# Compile Model Script for macOS
# =============================================================================

CLI_VENV="${1:-mlc-cli-venv}"
MODEL_PATH="${2:-}"
QUANTIZATION="${3:-q4f16_1}"
DEVICE="${4:-metal}"
OUTPUT_PATH="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RED='\033[1;31m'
GREEN='\033[0;32m'
BLUE='\033[1;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# Validate Inputs
# =============================================================================

if [[ -z "${MODEL_PATH}" ]]; then
    log_error "Model path is required"
fi

if [[ -z "${DEVICE}" ]]; then
    log_error "Device is required"
fi

if [[ -z "${OUTPUT_PATH}" ]]; then
    log_error "Output path is required"
fi

# =============================================================================
# Activate Environment
# =============================================================================

if ! command -v conda &> /dev/null; then
    log_error "Conda is not installed"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${CLI_VENV})"; then
    log_error "Environment '${CLI_VENV}' not found. Please build first."
fi

conda activate "${CLI_VENV}"

# =============================================================================
# Compile Model
# =============================================================================

log_info "Compiling model..."
log_info "  Model: ${MODEL_PATH}"
log_info "  Quantization: ${QUANTIZATION}"
log_info "  Device: ${DEVICE}"
log_info "  Output: ${OUTPUT_PATH}"

# Ensure output directory exists
mkdir -p "$(dirname "${OUTPUT_PATH}")"

# Run compilation
mlc_llm compile \
    "${MODEL_PATH}" \
    --quantization "${QUANTIZATION}" \
    --device "${DEVICE}" \
    -o "${OUTPUT_PATH}"

if [[ $? -eq 0 ]]; then
    log_success "Model compiled successfully: ${OUTPUT_PATH}"
else
    log_error "Model compilation failed"
fi

popd
conda deactivate
