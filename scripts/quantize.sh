#!/usr/bin/env bash
set -eu

# =============================================================================
# Quantize Model Script
# =============================================================================

CLI_VENV="${1:-mlc-cli-venv}"
MODEL_PATH="${2:-}"
QUANTIZATION="${3:-q4f16_1}"
OUTPUT_PATH="${4:-}"
CONV_TEMPLATE="${5:-llama-3}"

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

if [[ -z "${OUTPUT_PATH}" ]]; then
    # Generate default output path
    MODEL_NAME=$(basename "${MODEL_PATH}")
    OUTPUT_PATH="dist/${MODEL_NAME}-${QUANTIZATION}-MLC"
    log_info "Using default output path: ${OUTPUT_PATH}"
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
# Quantize Weights
# =============================================================================

log_info "Quantizing model weights..."
log_info "  Model: ${MODEL_PATH}"
log_info "  Quantization: ${QUANTIZATION}"
log_info "  Output: ${OUTPUT_PATH}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

# Convert weights
python -m mlc_llm convert_weight \
    "${MODEL_PATH}" \
    --quantization "${QUANTIZATION}" \
    -o "${OUTPUT_PATH}"

if [[ $? -ne 0 ]]; then
    log_error "Weight conversion failed"
fi

# =============================================================================
# Generate Config
# =============================================================================

log_info "Generating model config..."

python -m mlc_llm gen_config \
    "${MODEL_PATH}" \
    --quantization "${QUANTIZATION}" \
    --conv-template "${CONV_TEMPLATE}" \
    -o "${OUTPUT_PATH}"

if [[ $? -ne 0 ]]; then
    log_error "Config generation failed"
fi

log_success "Quantization completed: ${OUTPUT_PATH}"
log_info ""
log_info "Next steps:"
log_info "  1. Compile the model: ./mlc-cli compile --model ${OUTPUT_PATH}"
log_info "  2. Run the model: ./mlc-cli run --model-name $(basename ${OUTPUT_PATH})"

popd
conda deactivate
