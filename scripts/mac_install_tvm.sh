#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/versions.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"

source "$(conda info --base)/etc/profile.d/conda.sh"

# Args
CLI_VENV="${1:-mlc-cli-venv}"

if ! conda env list | awk '{print $1}' | grep -qx "${CLI_VENV}"; then
    conda create -n "${CLI_VENV}" -c "${CONDA_CHANNEL}" \
        "cmake>=${CMAKE_MIN_VERSION}" \
        rust \
        git \
        "${PYTHON_ABI_SPEC}" \
        psutil \
        pip
fi

conda activate "${CLI_VENV}"

# Verify Python version matches wheel requirement (from versions.sh)
PY_VERSION_INSTALLED=$(python --version | awk '{print $2}' | cut -d. -f1,2)
if [ "$PY_VERSION_INSTALLED" != "${PYTHON_VERSION}" ]; then
    echo "Error: mlc-cli-venv has Python $PY_VERSION_INSTALLED but wheels require Python ${PYTHON_VERSION}"
    echo "Recreating environment with correct Python version..."
    conda deactivate
    conda env remove -n "${CLI_VENV}" -y
    conda create -n "${CLI_VENV}" -c "${CONDA_CHANNEL}" \
        "cmake>=${CMAKE_MIN_VERSION}" \
        rust \
        git \
        "${PYTHON_ABI_SPEC}" \
        psutil \
        pip -y
    conda activate "${CLI_VENV}"
fi

# Install pre-built wheels from wheels directory
python -m pip install --force-reinstall "${WHEELS_DIR}"/tvm-*.whl
