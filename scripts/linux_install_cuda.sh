#!/bin/bash
# =============================================================================
# linux_install_cuda.sh — One-time CUDA toolkit setup helper
# =============================================================================
#
# WARNING: The CUDA version (13.0.2) and Ubuntu version (24.04) are pinned
# directly in this script. This is intentional — this is a one-time host
# setup helper, not part of the repeatable build pipeline.
#
# To use a different CUDA version or Ubuntu release:
#   1. Visit https://developer.nvidia.com/cuda-downloads
#   2. Select your OS, architecture, and installer type to get updated URLs.
#   3. Replace the .deb filename, wget URL, and cuda-toolkit-X-Y package name
#      below to match the version you need.
#   4. Ensure your GPU driver supports the chosen CUDA toolkit version.
# =============================================================================

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
sudo apt-get install -y nvidia-open