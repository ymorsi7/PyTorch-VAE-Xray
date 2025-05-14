#!/bin/bash

# Exit on error
set -e

# Activate virtual environment
source venv/bin/activate || (echo "Creating virtual environment" && python3 -m venv venv && source venv/bin/activate)

# Install requirements
pip install -r ignite_requirements.txt

# Train the improved model - will use our updated configuration
echo "Starting training of improved model with higher resolution and better detail preservation..."

# Run training and capture exit code
python chest_xray_vae.py
train_status=$?

# If training failed, check if there are checkpoints to continue from
if [ $train_status -ne 0 ]; then
    echo "Training encountered an error. Checking for previous checkpoints..."
    if [ -d "logs/ChestXray_DetailedMSSIMVAE" ]; then
        echo "Found previous training directory. Will try to resume from checkpoint."
        python chest_xray_vae.py
    else
        echo "No previous checkpoints found. Please check the error and try again."
        exit 1
    fi
fi

echo "Training complete! The improved model will be saved in logs/ChestXray_DetailedMSSIMVAE/"
echo "After training completes, you can generate images with:"
echo "./run_with_venv.sh" 