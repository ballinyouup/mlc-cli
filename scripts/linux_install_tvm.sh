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
        "${PYTHON_PACKAGE_SPEC}" \
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
        "${PYTHON_PACKAGE_SPEC}" \
        "${PYTHON_ABI_SPEC}" \
        pip \
        pytest \
        psutil
    conda activate "${CLI_VENV}"
fi

# Install the pre-built TVM wheel, then restore the bundled TVM FFI wheel.
# TVM's metadata only says "apache-tvm-ffi" without pinning the exact bundled source.
# Reinstalling the bundled FFI wheel after TVM prevents pip from leaving an incompatible PyPI FFI package.
mapfile -t TVM_FFI_WHEELS < <(
    find "${WHEELS_DIR}" -maxdepth 1 -type f \
        \( -name "apache_tvm_ffi-*.whl" \
           -o -name "apache-tvm-ffi-*.whl" \) \
        | sort
)

if [[ ${#TVM_FFI_WHEELS[@]} -eq 0 ]]; then
    echo "No bundled apache-tvm-ffi wheel found in ${WHEELS_DIR}"
    exit 1
fi

if [[ ${#TVM_FFI_WHEELS[@]} -gt 1 ]]; then
    printf '%s\n' "${TVM_FFI_WHEELS[@]}"
    echo "Multiple apache-tvm-ffi wheels found. Remove stale wheels and retry."
    exit 1
fi

mapfile -t TVM_WHEELS < <(
    find "${WHEELS_DIR}" -maxdepth 1 -type f \
        \( -name "tvm-*-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-*.whl" \
           -o -name "tvm-*-py3-none-*.whl" \) \
        | sort
)

if [[ ${#TVM_WHEELS[@]} -eq 0 ]]; then
    echo "No ABI-matching TVM wheel found in ${WHEELS_DIR}"
    exit 1
fi

if [[ ${#TVM_WHEELS[@]} -gt 1 ]]; then
    printf '%s\n' "${TVM_WHEELS[@]}"
    echo "Multiple ABI-matching TVM wheels found. Remove stale wheels and retry."
    exit 1
fi

python -m pip install --force-reinstall "${TVM_WHEELS[0]}"
python -m pip install --force-reinstall --no-deps "${TVM_FFI_WHEELS[0]}"
