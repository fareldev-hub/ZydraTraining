#!/usr/bin/env bash
echo "Running installation script..."
pip install -r requirements.txt

echo "Installation complete."
echo "Starting training script..."
python train.py