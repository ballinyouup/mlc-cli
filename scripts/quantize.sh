#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <MODEL_PATH> <QUANT_TYPE> <OUTPUT_DIR>"
    exit 1
fi

MODEL_PATH=$1
QUANT_TYPE=$2
OUTPUT_DIR=$3

echo "🚀 Starting Quantization: $QUANT_TYPE..."

CONDA_RUN="conda run --no-capture-output -n mlc-env python -m mlc_llm"

echo "📦 Compressing weights..."
$CONDA_RUN convert_weight "$MODEL_PATH" \
    --quantization "$QUANT_TYPE" \
    -o "$OUTPUT_DIR"

echo "📄 Generating chat config..."
$CONDA_RUN gen_config "$MODEL_PATH" \
    --quantization "$QUANT_TYPE" \
    --conv-template llama-3 \
    -o "$OUTPUT_DIR"

echo "✅ Quantization complete! Model saved to $OUTPUT_DIR"