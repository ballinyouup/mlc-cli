#!/usr/bin/env bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
CLI_VENV="${1:-mlc-cli-venv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"

if [ ! -d "${WHEELS_DIR}" ]; then
  echo "Wheels directory not found: ${WHEELS_DIR}"
  exit 1
fi

shopt -s nullglob
WHEELS=("${WHEELS_DIR}"/*.whl)
shopt -u nullglob

if [ ${#WHEELS[@]} -eq 0 ]; then
  echo "No .whl files found in: ${WHEELS_DIR}"
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${CLI_VENV}"; then
  echo "Conda env mlc-cli-venv already exists; reusing it."
else
  conda create -y -n mlc-cli-venv -c conda-forge python=3.13 pip
fi

conda activate "${CLI_VENV}"

python -m pip install --upgrade pip
python -m pip install "${WHEELS[@]}"