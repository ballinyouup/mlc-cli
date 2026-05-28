#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/versions.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"

source "$(conda info --base)/etc/profile.d/conda.sh"

# Args
CLI_VENV="${1:-mlc-cli-venv}"

# Python version must match the build env (mlc-build-venv) — see scripts/config/versions.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CLI_VENV}"; then
    conda create -y -n "${CLI_VENV}" -c "${CONDA_CHANNEL}" \
        "cmake>=${CMAKE_MIN_VERSION}" \
        rust \
        git \
        "${PYTHON_ABI_SPEC}" \
        pip \
        pytest \
        psutil
fi

conda activate "${CLI_VENV}"

# Check if Python version is correct, recreate if not
PY_VERSION_INSTALLED=$(python --version | awk '{print $2}' | cut -d. -f1,2)
if [ "$PY_VERSION_INSTALLED" != "${PYTHON_VERSION}" ]; then
    echo "Warning: Environment has Python $PY_VERSION_INSTALLED, but Python ${PYTHON_VERSION} is required. Recreating..."
    conda deactivate
    conda env remove -n "${CLI_VENV}" -y
    conda create -y -n "${CLI_VENV}" -c "${CONDA_CHANNEL}" \
        "cmake>=${CMAKE_MIN_VERSION}" \
        rust \
        git \
        "${PYTHON_ABI_SPEC}" \
        pip \
        pytest \
        psutil
    conda activate "${CLI_VENV}"
fi

# Install pre-built TVM wheel if one was built (relax/custom TVM source modes).
# In bundled mode linux_build_mlc.sh does not produce a standalone TVM wheel;
# TVM ships inside the mlc_llm wheel instead, so this step is a no-op there.
TVM_WHEELS=("${WHEELS_DIR}"/tvm-*.whl)
if [[ -f "${TVM_WHEELS[0]}" ]]; then
    python -m pip install --force-reinstall "${TVM_WHEELS[0]}"
else
    echo "No standalone TVM wheel found in ${WHEELS_DIR} (bundled mode — skipping TVM wheel install)"
fi
