#!/bin/bash
#
# Automated set-up script for christmAIs for any workspace
#
# Usage
# -----
# To use this script, simply run `./install-christmais.sh`
#

# Exit on error
set -e

finish() {
  if (( $? != 0)); then
    echo ""
    echo "================================================"
    echo "christmAIs did not install successfully"
    echo "Please refer to manual setup instructions:"
    echo "https://github.com/thinkingmachines/christmAIs"
    echo "================================================"
    echo ""
  fi
}
trap finish EXIT

# For printing error messages
err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
  exit 1
}

# Download important files
echo ""
echo "============================================"
echo "Downloading model checkpoint, categories, and"
echo "webdriver (chromedriver)                    "
echo "============================================"
echo ""

wget https://storage.googleapis.com/tm-christmais/categories.txt && \
    mkdir categories && \
    mv categories.txt categories/

wget https://storage.googleapis.com/download.magenta.tensorflow.org/models/arbitrary_style_transfer.tar.gz && \
    mkdir ckpt && \
    tar --strip-components 1 -xvzf arbitrary_style_transfer.tar.gz -C ckpt/

wget https://chromedriver.storage.googleapis.com/2.44/chromedriver_linux64.zip && \
    mkdir webdriver && \
    unzip chromedriver_*.zip -d webdriver

echo ""
echo "============================================"
echo "Download success! The following files are   "
echo "now stored in filesystem:                   "
echo "- categories: ./categories/categories.txt   "
echo "- model.ckpt: ./ckpt/model.ckpt             "
echo "- chromedriver: ./webdriver/chromedriver    "
echo "============================================"
echo ""

# Install rtmidi for realtime midi IO
if [[ $(which apt-get) ]]; then
    echo ""
    echo "============================================"
    echo "installing rtmidi Linux library dependencies"
    echo "sudo privileges required"
    echo "============================================"
    echo ""
    sudo apt-get install build-essential libasound2-dev libjack-dev python3-dev
fi
pip install --pre python-rtmidi

# Set up the magenta dependency
echo ""
echo "=============================="
echo "installing magenta dependency"
echo "=============================="
echo ""
pip install magenta

# Clone christmAIs repository
echo ""
echo "=============================="
echo "cloning christmAIs repository"
echo "=============================="
echo ""

python setup.py install

echo ""
echo "=============================="
echo "christmAIs Install Success!"
echo "=============================="
echo ""
