#!/bin/bash
set -e  # Exit on error

source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept build environment name as argument
BUILD_VENV="${1:-mlc-llm-venv}"
CLEAN_BUILD="${2:-no}"

# Check if mlc-llm directory exists
if [ ! -d "mlc-llm" ]; then
    echo "Error: mlc-llm directory not found. Please clone the repository first."
    exit 1
fi

conda activate ${BUILD_VENV}

# Update to latest mlc-llm and TVM-Unity to get bug fixes
echo "Updating mlc-llm and TVM-Unity to latest..."
cd mlc-llm
git pull origin main
git submodule update --init --recursive
cd ..

# Clean build removes old artifacts (needed when TVM has breaking changes)
if [ "$CLEAN_BUILD" = "yes" ]; then
    echo "Cleaning old build artifacts..."
    rm -rf mlc-llm/build
fi

NCORES=$(sysctl -n hw.ncpu)
mkdir -p mlc-llm/build
cd mlc-llm/build

cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . --parallel ${NCORES}

conda deactivate

