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
    SHORT_SHA="${MLC_LLM_REF:0:8}"
    MLC_WHEELS=($(ls "${WHEELS_DIR}"/mlc_llm-*-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-*.whl 2>/dev/null || true))
    SHA_MATCHES=()
    if [[ -n "${SHORT_SHA}" ]]; then
        SHA_MATCHES=($(printf '%s\n' "${MLC_WHEELS[@]}" | grep "g${SHORT_SHA}" || true))
    fi

    if [[ -n "${SHORT_SHA}" ]] && [[ ${#SHA_MATCHES[@]} -eq 1 ]]; then
        MLC_WHEEL_PATH="${SHA_MATCHES[0]}"
        echo "Selected MLC wheel matching MLC_LLM_REF short SHA (g${SHORT_SHA}): $(basename "${MLC_WHEEL_PATH}")"
    elif [[ -n "${SHORT_SHA}" ]] && [[ ${#SHA_MATCHES[@]} -gt 1 ]]; then
        echo "Error: Multiple MLC wheels match short SHA g${SHORT_SHA} in ${WHEELS_DIR}. Remove stale wheels and retry."
        printf '  %s\n' "${SHA_MATCHES[@]}" >&2
        exit 1
    elif [[ ${#MLC_WHEELS[@]} -eq 1 ]]; then
        MLC_WHEEL_PATH="${MLC_WHEELS[0]}"
        if [[ -n "${SHORT_SHA}" ]]; then
            echo "Warning: No MLC wheel matches MLC_LLM_REF short SHA (g${SHORT_SHA}). Using only available ABI-matching wheel: $(basename "${MLC_WHEEL_PATH}")"
        fi
    elif [[ ${#MLC_WHEELS[@]} -gt 1 ]]; then
        if [[ -n "${SHORT_SHA}" ]]; then
            echo "Error: Multiple ABI-matching MLC wheels exist and none matches MLC_LLM_REF short SHA (g${SHORT_SHA}). Delete stale wheels."
        else
            echo "Error: Multiple ABI-matching MLC wheels exist. Delete stale wheels."
        fi
        printf '  %s\n' "${MLC_WHEELS[@]}" >&2
        exit 1
    else
        echo "Error: No wheel found matching pattern: mlc_llm-*-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-*.whl (WHEELS_DIR=${WHEELS_DIR}, PYTHON_CP_TAG=${PYTHON_CP_TAG})"
        exit 1
    fi
    python -m pip install --force-reinstall "${MLC_WHEEL_PATH}"
else
    cd mlc-llm/python
    python -m pip install -e .
    cd ../..
fi
