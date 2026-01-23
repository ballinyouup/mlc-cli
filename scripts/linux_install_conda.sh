#!/bin/bash
set -e  # Exit on error

echo "Installing Miniconda for Linux..."

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
elif [ "$ARCH" = "aarch64" ]; then
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Download Miniconda installer
INSTALLER="/tmp/miniconda_installer.sh"
echo "Downloading Miniconda from $MINICONDA_URL..."
wget -O "$INSTALLER" "$MINICONDA_URL"

# Run installer
echo "Running Miniconda installer..."
bash "$INSTALLER" -b -p "$HOME/miniconda3"

# Clean up
rm "$INSTALLER"

# Initialize conda
echo "Initializing conda..."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
conda tos accept

echo "Miniconda installation complete!"
echo "Please restart your terminal or run: source ~/.bashrc"
