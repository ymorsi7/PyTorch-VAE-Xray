#!/bin/bash

echo "Cleaning up space before training..."

# Create backup directory for important checkpoints
mkdir -p checkpoint_backup

# Save only the latest checkpoint from each experiment
echo "Saving latest checkpoints..."
find logs -name "last.ckpt" -exec cp {} checkpoint_backup/ \;

# Remove older checkpoint versions
echo "Removing old logs and checkpoints..."
rm -rf logs/ChestXray_MSSIMVAE/version_[0-3]
rm -rf logs/ChestXray_DetailedMSSIMVAE/version_[0-1]

# Clean up any remaining temporary files
find . -name "*.tmp" -delete
find . -name "*.pyc" -delete

# Clear generated images if needed
if [ -d "generated_images" ]; then
    # Save a small subset of images
    mkdir -p backup_images
    find generated_images -name "*.png" | head -10 | xargs -I{} cp {} backup_images/
    
    # Remove the rest
    rm -rf generated_images
    mkdir -p generated_images
    
    # Restore backup
    cp backup_images/* generated_images/
fi

echo "Done cleaning up. Space usage after cleanup:"
du -h -d 1 .

echo "Disk space available:"
df -h | grep "System/Volumes/Data" 