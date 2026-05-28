#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/versions.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"

source "$(conda info --base)/etc/profile.d/conda.sh"

# Args
CLI_VENV="${1:-mlc-cli-venv}"
TVM_SOURCE="${2:-bundled}"  # bundled, relax, or custom
INSTALL_MODE="${3:-wheel}"  # source (editable from repo) or wheel (pre-built)

if ! conda env list | awk '{print $1}' | grep -qx "${CLI_VENV}"; then
    conda create -n "${CLI_VENV}" -c "${CONDA_CHANNEL}" \
        "cmake>=${CMAKE_MIN_VERSION}" \
        rust \
        git \
        "${PYTHON_PACKAGE_SPEC}" \
        "${PYTHON_ABI_SPEC}" \
        psutil \
        pip
fi

conda activate "${CLI_VENV}"

# Verify Python version matches wheel requirement (from versions.sh)
PY_VERSION_INSTALLED=$(python --version | awk '{print $2}' | cut -d. -f1,2)
if [ "$PY_VERSION_INSTALLED" != "${PYTHON_VERSION}" ]; then
    echo "Error: mlc-cli-venv has Python $PY_VERSION_INSTALLED but wheel requires Python ${PYTHON_VERSION}"
    echo "Recreating environment with correct Python version..."
    conda deactivate
    conda env remove -n "${CLI_VENV}" -y
    conda create -n "${CLI_VENV}" -c "${CONDA_CHANNEL}" \
        "cmake>=${CMAKE_MIN_VERSION}" \
        rust \
        git \
        "${PYTHON_PACKAGE_SPEC}" \
        "${PYTHON_ABI_SPEC}" \
        psutil \
        pip -y
    conda activate "${CLI_VENV}"
fi

# Install TVM wheel first (MLC depends on TVM at runtime)
if ls "${WHEELS_DIR}"/tvm-*.whl 1>/dev/null 2>&1; then
    echo "Installing TVM wheel (dependency for MLC)..."
    python -m pip install --force-reinstall "${WHEELS_DIR}"/tvm-*.whl
else
    echo "Warning: No TVM wheel found in ${WHEELS_DIR}. MLC may fail if TVM is not already installed."
fi

# install MLC Python package
if [ "${INSTALL_MODE}" = "wheel" ]; then
    python -m pip install --force-reinstall "${WHEELS_DIR}"/mlc_llm-*.whl
else
    cd mlc-llm/python
    python -m pip install -e .
    cd ../..
fi
