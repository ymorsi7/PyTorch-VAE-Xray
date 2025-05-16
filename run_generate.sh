#!/bin/bash
source venv/bin/activate
python evaluate_vae.py --checkpoint logs/ChestXray_MSSIMVAE/version_*/checkpoints/last.ckpt --output_dir generated_images --num_samples 100 --device mps 