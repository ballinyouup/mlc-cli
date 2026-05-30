#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/versions.sh"
source "${SCRIPT_DIR}/lib/wheel_selection.sh"

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
        pytest \
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
        pytest \
        psutil \
        pip -y
    conda activate "${CLI_VENV}"
fi

# Install TVM wheel first (MLC depends on TVM at runtime)
TVM_WHEELS=($(ls "${WHEELS_DIR}"/tvm-*-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-*.whl 2>/dev/null || true))
if [ ${#TVM_WHEELS[@]} -gt 1 ]; then
    echo "Warning: Multiple TVM wheels found. Using the first one: $(basename "${TVM_WHEELS[0]}")"
fi
if [ ${#TVM_WHEELS[@]} -gt 0 ]; then
    echo "Installing TVM wheel (dependency for MLC)..."
    python -m pip install --force-reinstall "${TVM_WHEELS[0]}"
else
    echo "Warning: No TVM wheel found in ${WHEELS_DIR} matching tvm-*-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-*.whl. MLC may fail if TVM is not already installed."
fi

# install MLC Python package
if [ "${INSTALL_MODE}" = "wheel" ]; then
    find_mlc_wheel
    if [[ -n "${MLC_WHEEL_PATH}" ]]; then
        python -m pip install --force-reinstall "${MLC_WHEEL_PATH}"
    fi
else
    cd mlc-llm/python
    python -m pip install -e .
    cd ../..
fi
