import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.utils as vutils
import os
from pathlib import Path

# Import the dataset class from your existing code
from chest_xray_vae import ChestXrayDataset

# Define the same transforms used in your data loading pipeline
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])

# Create the dataset objects
train_dataset = ChestXrayDataset(
    data_path=".",
    split='train',
    transform=transform,
)

# Get images from both classes (normal and pneumonia)
normal_images = []
pneumonia_images = []
normal_count = 0
pneumonia_count = 0

# We need 36 images of each class (for a total of 72 in a 6x12 grid)
target_per_class = 36

# Collect images
for i in range(len(train_dataset)):
    img, label = train_dataset[i]
    
    if label == 0 and normal_count < target_per_class:  # Normal
        normal_images.append(img)
        normal_count += 1
    elif label == 1 and pneumonia_count < target_per_class:  # Pneumonia
        pneumonia_images.append(img)
        pneumonia_count += 1
        
    # Break if we have enough images
    if normal_count >= target_per_class and pneumonia_count >= target_per_class:
        break

# Combine images from both classes
all_images = normal_images + pneumonia_images

# Convert to a tensor
all_images = torch.stack(all_images)

# Normalize images to [0, 1] range for proper display
if all_images.min() < 0:
    all_images = (all_images + 1) / 2.0

# Create the grid (6 rows x 12 columns)
grid = vutils.make_grid(all_images, nrow=12, padding=2, normalize=False)

# Convert to image
plt.figure(figsize=(24, 12))
plt.axis('off')
plt.title('Chest X-Ray Dataset Samples (Top: Normal, Bottom: Pneumonia)', fontsize=20)
plt.imshow(grid.permute(1, 2, 0))

# Save with high resolution
os.makedirs('report_images', exist_ok=True)
plt.savefig('report_images/chest_xray_samples.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
print("Image grid saved to 'report_images/chest_xray_samples.png'") 