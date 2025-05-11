import os
import glob
from pathlib import Path

def find_checkpoints():
    """Find all checkpoint files in the logs directory"""
    logs_dir = Path("logs")
    
    if not logs_dir.exists():
        print("No logs directory found. You need to train the model first.")
        return []
    
    checkpoints = []
    
    # Find all checkpoint files
    for checkpoint in logs_dir.glob("**/checkpoints/*.ckpt"):
        checkpoints.append(str(checkpoint))
    
    return checkpoints

if __name__ == "__main__":
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found. You need to train the model first.")
        print("\nTo train the model, run one of the following commands:")
        print("  ./train_with_venv.sh")
        print("  or")
        print("  python chest_xray_vae.py")
    else:
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for i, cp in enumerate(checkpoints):
            print(f"{i+1}. {cp}")
        
        print("\nTo evaluate a checkpoint, run:")
        print(f"python evaluate_vae.py --checkpoint \"{checkpoints[0]}\" --output_dir generated_images")
        print("or with the virtual environment:")
        print(f"./run_with_venv.sh --checkpoint \"{checkpoints[0]}\" --output_dir generated_images") 