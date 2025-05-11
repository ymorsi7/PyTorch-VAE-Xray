#!/bin/bash

# Exit on error
set -e

# Print commands
set -x

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r ignite_requirements.txt
pip install -r requirements.txt

# List installed packages (for debugging)
pip list | grep -E 'torch|pytorch|ignite'

# Run the training script
python chest_xray_vae.py

# Alternatively, you can use the original training script with our config
# python run.py -c configs/chest_xray_mssim_vae.yaml

# Deactivate virtual environment
deactivate

echo "Training complete. Check logs/ directory for results." 