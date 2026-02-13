set -e 

ENV_NAME="${1:-mlc-cli-venv}"
MODEL_PATH="$2"
QUANT_TYPE="$3"
OUTPUT_DIR="$4"
CONV_TEMPLATE="$5"

if [ -z "$MODEL_PATH" ]; then
    echo "🔮 Interactive Mode (No arguments provided)"
    read -p "Enter CLI Environment Name [mlc-cli-venv]: " input_env
    ENV_NAME="${input_env:-mlc-cli-venv}"
    
    read -p "Enter Model Path: " MODEL_PATH
    
    echo "Select Quantization:"
    select QUANT_TYPE in "q4f16_1" "q3f16_1" "q0f16"; do
        [ -n "$QUANT_TYPE" ] && break
    done
    
    read -p "Enter Output Directory [dist/model-output]: " input_out
    OUTPUT_DIR="${input_out:-dist/model-output}"
    
    echo "Select Template:"
    select CONV_TEMPLATE in "llama-3" "chatml" "mistral_default" "phi-2" "gemma" "qwen2"; do
        [ -n "$CONV_TEMPLATE" ] && break
    done
fi

if [ -z "$MODEL_PATH" ] || [ -z "$QUANT_TYPE" ] || [ -z "$CONV_TEMPLATE" ]; then
    echo "❌ Error: Missing required arguments."
    echo "Usage: $0 [ENV] [MODEL] [QUANT] [OUTPUT] [TEMPLATE]"
    exit 1
fi

echo "🚀 Starting Quantization: $QUANT_TYPE..."


CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "⚠️  Could not find conda.sh! Assuming conda is already in PATH."
fi

echo "🔌 Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

echo "📦 Compressing weights..."
python -m mlc_llm convert_weight "$MODEL_PATH" \
    --quantization "$QUANT_TYPE" \
    -o "$OUTPUT_DIR"

echo "📄 Generating chat config..."
python -m mlc_llm gen_config "$MODEL_PATH" \
    --quantization "$QUANT_TYPE" \
    --conv-template "$CONV_TEMPLATE" \
    -o "$OUTPUT_DIR"

echo "🔌 Deactivating environment..."
conda deactivate

echo "✅ Quantization complete! Model saved to $OUTPUT_DIR"