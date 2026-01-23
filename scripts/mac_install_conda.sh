#!/bin/bash
set -e  # Exit on error

echo "Installing Miniconda for macOS..."

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
else
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
fi

# Download Miniconda installer
INSTALLER="/tmp/miniconda_installer.sh"
echo "Downloading Miniconda from $MINICONDA_URL..."
curl -L -o "$INSTALLER" "$MINICONDA_URL"

# Run installer
echo "Running Miniconda installer..."
bash "$INSTALLER" -b -p "$HOME/miniconda3"

# Clean up
rm "$INSTALLER"

# Initialize conda
echo "Initializing conda..."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init zsh bash
conda tos accept

echo "Miniconda installation complete!"
echo "Please restart your terminal or run: source ~/.zshrc (or ~/.bashrc)"
