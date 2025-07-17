#!/bin/bash
echo "Installing CPU-only PyTorch..."
pip install torch==2.2.2 --index-url=https://download.pytorch.org/whl/cpu

echo "Installing other dependencies..."
pip install -r requirements.txt
