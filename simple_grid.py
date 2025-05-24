from PIL import Image
import os
from pathlib import Path

def create_image_grid(images, rows, cols, output_path):
    """Create a grid of images using only PIL"""
    # If image sizes vary, resize them
    width, height = images[0].size
    
    # Create a new image with the appropriate dimensions
    grid_width = cols * width + (cols - 1) * 10  # 10px padding between images
    grid_height = rows * height + (rows - 1) * 10
    grid_img = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    
    # Place images in the grid
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * (width + 10)
        y = row * (height + 10)
        grid_img.paste(img, (x, y))
    
    # Save the grid
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid_img.save(output_path, quality=95)
    print(f"Grid saved to {output_path}")

# Load generated images
generated_dir = Path("generated_images")
generated_images = []

# Get first 36 generated images
for path in sorted(generated_dir.glob("generated_*.png")):
    if "grid" not in str(path) and len(generated_images) < 36:
        img = Image.open(path)
        # Resize to 128x128 for better visibility
        img = img.resize((128, 128), Image.BICUBIC)
        generated_images.append(img)

# Create and save the grid (6x6)
create_image_grid(generated_images, 6, 6, "report_images/simple_generated_grid.png")

# You can also create a side-by-side comparison of real vs generated
# Just load the real images with PIL and do the same 