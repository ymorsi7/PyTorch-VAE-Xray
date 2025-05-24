import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.utils as vutils
import os
from PIL import Image
from pathlib import Path

# Directory containing generated images
generated_dir = Path("generated_images")

# Load the first 72 generated images
images = []
count = 0
target_count = 72  # 6x12 grid

# Look for generated images
for img_path in sorted(generated_dir.glob("generated_*.png")):
    if "grid" not in str(img_path):  # Skip grid images
        # Load image
        img = Image.open(img_path)
        # Convert to tensor
        img_tensor = transforms.ToTensor()(img)
        images.append(img_tensor)
        count += 1
        
        if count >= target_count:
            break

# Convert to a tensor
all_images = torch.stack(images)

# Create the grid (6 rows x 12 columns)
grid = vutils.make_grid(all_images, nrow=12, padding=2, normalize=False)

# Convert to image
plt.figure(figsize=(24, 12))
plt.axis('off')
plt.title('Generated Chest X-Ray Samples', fontsize=20)
plt.imshow(grid.permute(1, 2, 0))

# Save with high resolution
os.makedirs('report_images', exist_ok=True)
plt.savefig('report_images/generated_xray_samples.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
print("Image grid saved to 'report_images/generated_xray_samples.png'") 