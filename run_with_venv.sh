#!/bin/bash

# Exit on error
set -e

# Print commands
set -x

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r ignite_requirements.txt

# List installed packages (for debugging)
echo "Installed packages:"
pip list | grep -E 'torch|pytorch|ignite'

# Check if logs directory exists
if [ ! -d "logs" ]; then
    echo "No checkpoint found. Running training first..."
    python chest_xray_vae.py
fi

# Find a valid checkpoint - look for both the old and new naming convention
CHECKPOINT=$(python -c "
import glob
from pathlib import Path
# First look for our improved model checkpoints
improved_checkpoints = list(Path('logs').glob('ChestXray_DetailedMSSIMVAE/**/checkpoints/last.ckpt'))
# Then fall back to older model if needed
default_checkpoints = list(Path('logs').glob('**/checkpoints/last.ckpt'))
# Use improved if available, otherwise use default
if improved_checkpoints:
    print(improved_checkpoints[0])
elif default_checkpoints:
    print(default_checkpoints[0])
else:
    print('')
")

if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found after training. Running training first..."
    python chest_xray_vae.py
    
    # Try finding checkpoint again
    CHECKPOINT=$(python -c "
    import glob
    from pathlib import Path
    checkpoints = list(Path('logs').glob('**/checkpoints/last.ckpt'))
    print(checkpoints[0] if checkpoints else '')
    ")
    
    if [ -z "$CHECKPOINT" ]; then
        echo "No checkpoint found after training. Something went wrong."
        exit 1
    fi
fi

echo "Using checkpoint: $CHECKPOINT"

# Run the evaluation script with provided checkpoint
python evaluate_vae.py --checkpoint "$CHECKPOINT" --output_dir generated_images

# Deactivate virtual environment
deactivate

echo "Evaluation complete. Results saved to generated_images/" 