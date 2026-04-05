#!/usr/bin/env bash
set -eu

# =============================================================================
# Conda Installation Script for Linux
# =============================================================================

RED='\033[1;31m'
GREEN='\033[0;32m'
BLUE='\033[1;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# Check for existing installation
# =============================================================================

if command -v conda &> /dev/null; then
    log_info "Conda is already installed"
    conda --version
    exit 0
fi

log_info "Installing Miniforge (lightweight conda)..."

# =============================================================================
# Download and Install Miniforge
# =============================================================================

ARCH=$(uname -m)
case "${ARCH}" in
    x86_64)
        CONDA_ARCH="x86_64"
        ;;
    aarch64)
        CONDA_ARCH="aarch64"
        ;;
    *)
        log_error "Unsupported architecture: ${ARCH}"
        ;;
esac

MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${CONDA_ARCH}.sh"
INSTALLER_PATH="/tmp/miniforge_installer.sh"

log_info "Downloading from: ${MINIFORGE_URL}"
wget -q -O "${INSTALLER_PATH}" "${MINIFORGE_URL}"

log_info "Installing Miniforge..."
bash "${INSTALLER_PATH}" -b "$HOME/miniforge3" -p

# =============================================================================
# Post-install Setup
# =============================================================================

# Initialize conda in shell
CONDA_BASE="$HOME/miniforge3"
"${CONDA_BASE}/bin/conda" init bash

# Add to PATH for current session
export PATH="${CONDA_BASE}/bin:${PATH}"

# Cleanup
rm -f "${INSTALLER_PATH}"

log_success "Conda (Miniforge) installed successfully!"
log_info ""
log_info "Installation location: ${CONDA_BASE}"
log_info ""
log_info "To activate conda in your current terminal:"
log_info "  source ~/.bashrc"
log_info "  conda activate base"
log_info ""
log_info "Or restart your terminal."
