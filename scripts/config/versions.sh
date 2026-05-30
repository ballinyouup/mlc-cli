#!/usr/bin/env bash
# =============================================================================
# scripts/config/versions.sh — Central version/dependency configuration
# =============================================================================
# Source this file from any build or install script to get consistent values:
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/config/versions.sh"
#
# Do NOT hardcode Python versions, repo URLs, CUDA arch, or conda channel
# in individual scripts. Change them here instead.
#
# Matching Go constants live in versions_defaults.go — keep them in sync.
# =============================================================================

# -----------------------------------------------------------------------------
# Python version
# -----------------------------------------------------------------------------
# Must match the build environment (mlc-build-venv) so that produced cp* wheels
# are installable in the CLI environment (mlc-cli-venv) without ABI mismatch.
# To switch to a different Python version, change PYTHON_VERSION here only.
PYTHON_VERSION="3.13"

# Derived values — do not edit these; edit PYTHON_VERSION above.
# PYTHON_CP_TAG: e.g. "cp311" from "3.11"
PYTHON_CP_TAG="cp$(echo "${PYTHON_VERSION}" | tr -d '.')"

# PYTHON_PACKAGE_SPEC chooses the Python version.
PYTHON_PACKAGE_SPEC="python=${PYTHON_VERSION}"

# PYTHON_ABI_SPEC chooses the CPython ABI tag, e.g. cp311 or cp313.
# This prevents Python 3.13 from resolving to free-threading cp313t when we need cp313 wheels.
PYTHON_ABI_SPEC="python_abi=${PYTHON_VERSION}=*_${PYTHON_CP_TAG}"

# -----------------------------------------------------------------------------
# MLC-LLM repository
# -----------------------------------------------------------------------------
MLC_LLM_REPO="https://github.com/mlc-ai/mlc-llm"

# Known-good mlc-llm revision for reproducible builds.
# This SHA passed the Python 3.13 CUDA flow: build/install, import, quantize, compile, and run.
# Set empty only when intentionally testing the upstream default branch HEAD.
MLC_LLM_REF="2008fe8343e1f40ef89ee57b9287aebcf1b86c98"

# -----------------------------------------------------------------------------
# TVM / mlc-ai/relax repository (used in relax TVM source mode)
# -----------------------------------------------------------------------------
TVM_REPO="https://github.com/mlc-ai/relax.git"
TVM_REF="b628d91fac716679db539884a55f8c6651f54dea"   # known-good commit SHA

# -----------------------------------------------------------------------------
# Conda channel and cmake minimum version
# -----------------------------------------------------------------------------
CONDA_CHANNEL="conda-forge"

# Minimum cmake version required by mlc-llm's CMakeLists.
# Scripts use this as: "cmake>=${CMAKE_MIN_VERSION}"
# (The variable holds only the version number; the ">=" operator goes in each
# conda create call so the intent is unambiguous.)
CMAKE_MIN_VERSION="3.24"

# LLVM major version used for macOS TVM build dependency llvmdev.
LLVM_VERSION="19"

# Packages included in every conda create call.
# Scripts may append extra platform-specific packages after sourcing this file.
CONDA_BASE_PKGS=(
    "cmake>=${CMAKE_MIN_VERSION}"
    "rust"
    "git"
    "pip"
)

# -----------------------------------------------------------------------------
# CUDA default architecture  (Linux/CUDA only — ignored on macOS)
# -----------------------------------------------------------------------------
# SM86 covers: RTX 30xx consumer Ampere, A10, A30, A40, RTX A-series.
# NOTE: A100 is SM80, NOT SM86. H100 is SM90.
#
# This default targets RTX 30xx / A10. You MUST override it to match
# your actual GPU:
#   SM70 = V100          SM80 = A100       SM86 = RTX 30xx / A10
#   SM89 = RTX 40xx      SM90 = H100/H200
#
# Override via the CUDA_ARCH build argument or by changing this value.
CUDA_ARCH_DEFAULT="86"
