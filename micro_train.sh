#!/bin/bash

# Exit on error but not on disk full
set +e

# Activate virtual environment
source venv/bin/activate || (echo "Creating virtual environment" && python3 -m venv venv && source venv/bin/activate)

# Set up a temporary directory on external storage if possible
if [ -d "/Volumes/Transcend" ]; then
    # Use external drive for logs
    echo "Using external drive for logs"
    export TMPDIR="/Volumes/Transcend/temp_logs"
    mkdir -p $TMPDIR
    
    # Link logs directory to external drive
    if [ -d "logs" ]; then
        mv logs $TMPDIR/
        ln -s $TMPDIR/logs logs
    fi
fi

# Modify config for minimal disk usage
cat > minimal_config.py << EOF
def create_minimal_config():
    """Create a tiny config for testing with minimal disk usage"""
    config = {
        'model_params': {
            'name': 'MSSIMVAE',
            'in_channels': 3,
            'latent_dim': 256,  # Smaller latent dim
            'hidden_dims': [32, 64, 128, 256],  # Smaller model
            'loss_type': 'mssim',
            'alpha': 0.0015,
            'kernel_size': 4,
            'M_N': 0.0025,
        },
        'data_params': {
            'data_path': '.',
            'train_batch_size': 32,  # Smaller batch
            'val_batch_size': 32,
            'patch_size': 64,
            'num_workers': 2,  # Fewer workers
        },
        'exp_params': {
            'manual_seed': 1234,
            'LR': 0.002,
            'weight_decay': 0.00001,
            'scheduler_gamma': 0.97,
            'kld_weight': 0.00015,
            'min_delta': 0.01
        },
        'trainer_params': {
            'accelerator': 'auto',
            'devices': 1,
            'max_epochs': 20,  # Minimal epochs
            'gradient_clip_val': 1.0,
            'log_every_n_steps': 50,  # Minimal logging
            'enable_checkpointing': True,
            'enable_progress_bar': True,
            'limit_train_batches': 0.1,  # Use only 10% of data
            'limit_val_batches': 0.1,  # Use only 10% of data
        },
        'logging_params': {
            'save_dir': 'logs',
            'name': 'MicroVAE',  # New name
            'flush_logs_every_n_steps': 200,
            'log_save_interval': 10,
        }
    }
    return config
EOF

# Add import to Python file
grep -q "create_minimal_config" chest_xray_vae.py || cat >> chest_xray_vae.py << EOF

# Import minimal config for small disk usage
from minimal_config import create_minimal_config
EOF

# Run with the minimal config
echo "Starting micro training to save disk space..."
python -c "
import sys
import os
from chest_xray_vae import main
from minimal_config import create_minimal_config

# Monkey patch the config function
import chest_xray_vae
chest_xray_vae.create_config = create_minimal_config

# Run main
main()
"

echo "Micro training complete!" 