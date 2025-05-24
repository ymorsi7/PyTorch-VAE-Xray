import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.utils as vutils
import os
import numpy as np
from PIL import Image
from pathlib import Path

# Import the dataset class from your existing code
from chest_xray_vae import ChestXrayDataset

# Define transforms for real images
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])

# Create the dataset objects for real images
train_dataset = ChestXrayDataset(
    data_path=".",
    split='train',
    transform=transform,
)

# Get a sample of real images (just using the first 10 for demonstration)
real_images = []
for i in range(10):
    img, _ = train_dataset[i]
    real_images.append(img)

# Convert to tensor
real_images = torch.stack(real_images)

# Normalize to [0, 1] range if needed
if real_images.min() < 0:
    real_images = (real_images + 1) / 2.0

# Load 10 generated images
generated_dir = Path("generated_images")
gen_images = []
count = 0

# Look for generated images
for img_path in sorted(generated_dir.glob("generated_*.png")):
    if "grid" not in str(img_path) and count < 10:  # Skip grid images, only take 10
        # Load image
        img = Image.open(img_path)
        # Resize to match real images if needed
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.BICUBIC)
        # Convert to tensor
        img_tensor = transforms.ToTensor()(img)
        gen_images.append(img_tensor)
        count += 1

# Convert to tensor
gen_images = torch.stack(gen_images)

# Create figure for comparison
fig, axs = plt.subplots(10, 2, figsize=(8, 20))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i in range(10):
    # Display real image
    axs[i, 0].imshow(real_images[i].permute(1, 2, 0))
    axs[i, 0].axis('off')
    if i == 0:
        axs[i, 0].set_title('Real', fontsize=16)
    
    # Display generated image
    axs[i, 1].imshow(gen_images[i].permute(1, 2, 0))
    axs[i, 1].axis('off')
    if i == 0:
        axs[i, 1].set_title('Generated', fontsize=16)

# Save high-resolution comparison
os.makedirs('report_images', exist_ok=True)
plt.savefig('report_images/real_vs_generated.png', bbox_inches='tight', dpi=300)
print("Comparison saved to 'report_images/real_vs_generated.png'") 