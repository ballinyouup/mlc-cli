#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <ENV_NAME> <MODEL_PATH> <QUANT_TYPE> <OUTPUT_DIR> <CONV_TEMPLATE>"
    exit 1
fi

ENV_NAME=$1
MODEL_PATH=$2
QUANT_TYPE=$3
OUTPUT_DIR=$4
CONV_TEMPLATE=$5

echo "🚀 Starting Quantization: $QUANT_TYPE..."

CONDA_RUN="conda run --no-capture-output -n $ENV_NAME python -m mlc_llm"

echo "📦 Compressing weights..."
$CONDA_RUN convert_weight "$MODEL_PATH" \
    --quantization "$QUANT_TYPE" \
    -o "$OUTPUT_DIR"

echo "📄 Generating chat config..."
$CONDA_RUN gen_config "$MODEL_PATH" \
    --quantization "$QUANT_TYPE" \
    --conv-template "$CONV_TEMPLATE" \
    -o "$OUTPUT_DIR"

echo "✅ Quantization complete! Model saved to $OUTPUT_DIR"