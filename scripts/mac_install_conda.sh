#!/usr/bin/env bash
set -eu

# =============================================================================
# Conda Installation Script for macOS
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

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    log_error "Homebrew is required. Install from: https://brew.sh"
fi

log_info "Installing Miniforge via Homebrew..."

# =============================================================================
# Install via Homebrew
# =============================================================================

brew install --cask miniforge

# =============================================================================
# Post-install Setup
# =============================================================================

# Initialize conda
CONDA_BASE="/opt/homebrew/Caskroom/miniforge/base"

if [ -f "${CONDA_BASE}/bin/conda" ]; then
    "${CONDA_BASE}/bin/conda" init zsh  # macOS typically uses zsh
    log_success "Conda (Miniforge) installed successfully!"
    log_info ""
    log_info "Installation location: ${CONDA_BASE}"
    log_info ""
    log_info "To activate conda:"
    log_info "  source ~/.zshrc"
    log_info "  conda activate base"
else
    # Fallback: try to find conda
    log_info "Setting up conda..."
    "$(brew --prefix miniforge)/bin/conda" init zsh || true
    log_success "Conda installed. Restart your terminal to activate."
fi
