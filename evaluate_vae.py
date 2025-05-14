import os
import torch
import numpy as np
from pathlib import Path
import argparse
from torchvision import transforms, models
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import PIL.Image as Image
from scipy.linalg import sqrtm
from pytorch_lightning import seed_everything

# Ignite imports for FID and IS metrics
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore

from models import vae_models
from experiment import VAEXperiment
from chest_xray_vae import ChestXrayDataset, ChestXrayDataModule, create_config

def load_model(checkpoint_path):
    """Load the trained VAE model from checkpoint"""
    # Get the config
    config = create_config()
    
    # Create a placeholder model
    temp_model = vae_models[config['model_params']['name']](**config['model_params'])
    
    # Create experiment 
    experiment = VAEXperiment.load_from_checkpoint(
        checkpoint_path, 
        vae_model=temp_model,
        params=config['exp_params']
    )
    
    model = experiment.model
    model.eval()
    return model

def interpolate(batch):
    """Resize images to 299x299 for Inception model using PIL for better interpolation"""
    arr = []
    for img in batch:
        # Handle batch items that might be a list or tensor
        if isinstance(img, list) or len(img.shape) > 3:
            img = img[0]  # Just take the first item if it's a batch
            
        # Convert to PIL image and resize
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        
        # Convert back to tensor
        tensor_img = transforms.ToTensor()(resized_img)
        
        # Ensure it's a 3-channel image (RGB)
        if tensor_img.shape[0] == 1:  # If grayscale, repeat to make 3 channels
            tensor_img = tensor_img.repeat(3, 1, 1)
            
        arr.append(tensor_img)
    
    return torch.stack(arr)

def generate_samples(model, num_samples=1000, batch_size=64, device='cuda'):
    """Generate samples from the trained model"""
    model.to(device)
    model.eval()
    
    generated_images = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            z = torch.randn(current_batch_size, model.latent_dim).to(device)
            samples = model.decode(z)
            generated_images.append(samples.cpu())
    
    return torch.cat(generated_images, dim=0)

def save_generated_images(images, output_dir, prefix='generated', upscale_size=256):
    """Save generated images to disk with optional upscaling and enhancement"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Denormalize images from [-1,1] to [0,1] range if they were normalized
    if images.min() < 0:
        images = (images + 1) / 2.0
    
    # Apply enhancement to make details more visible
    enhanced_images = []
    for img in images:
        # Convert to PIL for enhancement
        pil_img = transforms.ToPILImage()(img)
        
        # Enhance contrast (make whites whiter and darks darker)
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(1.3)  # Increase contrast by 30%
        
        # Enhance sharpness
        sharpener = ImageEnhance.Sharpness(enhanced)
        enhanced = sharpener.enhance(1.5)  # Increase sharpness by 50%
        
        # Convert back to tensor
        enhanced_tensor = transforms.ToTensor()(enhanced)
        enhanced_images.append(enhanced_tensor)
    
    # Convert back to a batch tensor
    enhanced_images = torch.stack(enhanced_images)
    
    # Create a grid of sample images (for visualization)
    grid_img = vutils.make_grid(enhanced_images[:64], nrow=8, normalize=False, padding=2)
    vutils.save_image(grid_img, os.path.join(output_dir, f"{prefix}_grid.png"))
    
    # Upscale images before saving
    for i, img in enumerate(enhanced_images):
        img_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        
        # Convert to PIL, upscale, and save
        pil_img = transforms.ToPILImage()(img)
        pil_img = pil_img.resize((upscale_size, upscale_size), Image.BICUBIC)
        pil_img.save(img_path)
    
    # Also save an upscaled grid
    pil_grid = transforms.ToPILImage()(grid_img)
    grid_upscaled = pil_grid.resize((upscale_size * 8, upscale_size), Image.BICUBIC)
    grid_upscaled.save(os.path.join(output_dir, f"{prefix}_grid_large.png"))
    
    return os.path.join(output_dir, f"{prefix}_grid_large.png")

def calculate_fid_score(real_imgs, fake_imgs, device='cuda'):
    """Calculate FID score manually using scipy."""
    try:
        # Load the inception model for feature extraction
        inception = models.inception_v3(weights='DEFAULT', transform_input=False).to(device)
        inception.eval()
        
        # Remove the last classification layer
        inception.fc = torch.nn.Identity()
        
        # Extract features in evaluation mode
        def get_features(images, batch_size=32):
            features = []
            with torch.no_grad():
                for i in range(0, len(images), batch_size):
                    batch = images[i:i + batch_size]
                    batch_features = inception(batch)
                    features.append(batch_features.cpu().numpy())
            return np.concatenate(features)
        
        # Get features for real and fake images
        print("Extracting features from real images...")
        real_features = get_features(real_imgs)
        print("Extracting features from generated images...")
        fake_features = get_features(fake_imgs)
        
        # Calculate mean and covariance
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Calculate FID score
        diff = mu_real - mu_fake
        
        # Product might be almost singular
        covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        return float(fid)
    
    except Exception as e:
        print(f"Error calculating manual FID score: {str(e)}")
        return None

def evaluate_with_ignite_metrics(real_dataset, generated_images, device='cuda'):
    """Evaluate FID and Inception Score using PyTorch-Ignite metrics"""
    fid_score = None
    is_score = None
    
    try:
        # Process generated images for metrics
        print("Processing generated images...")
        processed_imgs = []
        for img in generated_images:
            # Convert to PIL for proper resizing
            pil_img = transforms.ToPILImage()(img)
            resized = transforms.Resize((299, 299))(pil_img)
            tensor = transforms.ToTensor()(resized)
            # Convert grayscale to RGB if needed
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            processed_imgs.append(tensor)
        
        # Convert to batch tensor
        fake_imgs = torch.stack(processed_imgs).to(device)
        
        # Process real images - extract first 500 for comparison
        print("Processing real images...")
        real_imgs = []
        for i in range(min(500, len(real_dataset))):
            img, _ = real_dataset[i]
            # Convert to PIL for proper resizing
            pil_img = transforms.ToPILImage()(img)
            resized = transforms.Resize((299, 299))(pil_img)
            tensor = transforms.ToTensor()(resized)
            # Convert grayscale to RGB if needed
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            real_imgs.append(tensor)
        
        # Convert to batch tensor
        real_imgs = torch.stack(real_imgs).to(device)
        
        # Calculate Inception Score first (only uses fake images)
        print("Calculating Inception Score...")
        inception = InceptionScore(device=device)
        try:
            # Process in smaller batches to avoid CUDA memory issues
            batch_size = 32
            for i in range(0, len(fake_imgs), batch_size):
                batch = fake_imgs[i:i+batch_size]
                inception.update(batch)
            
            # Get the score
            score = inception.compute()
            if isinstance(score, torch.Tensor):
                is_score = score.item()
            else:
                is_score = float(score)
            print(f"Inception Score: {is_score:.4f}")
        except Exception as e:
            print(f"Error calculating Inception Score: {str(e)}")
        
        # Calculate FID Score directly instead of using Ignite
        print("Calculating FID Score...")
        fid_score = calculate_fid_score(real_imgs, fake_imgs, device)
        if fid_score is not None:
            print(f"FID Score: {fid_score:.4f}")
        
    except Exception as e:
        print(f"General error in metrics calculation: {str(e)}")
    
    return fid_score, is_score

def main():
    parser = argparse.ArgumentParser(description='Evaluate VAE model and generate images')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='generated_images',
                       help='Directory to save generated images')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint)
    model.to(args.device)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    generated_images = generate_samples(model, num_samples=args.num_samples, device=args.device)
    
    # Save generated images
    print(f"Saving generated images to {args.output_dir}")
    grid_path = save_generated_images(generated_images, args.output_dir)
    print(f"Sample grid saved to {grid_path}")
    
    # Prepare real dataset for comparison
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    
    # Load real images
    real_dataset = ChestXrayDataset(
        data_path=".",
        split="test",
        transform=transform
    )
    
    # Calculate FID and IS scores using PyTorch-Ignite
    print("Calculating FID and Inception Score using PyTorch-Ignite...")
    fid_score, is_score = evaluate_with_ignite_metrics(real_dataset, generated_images, device=args.device)
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        if fid_score is not None:
            f.write(f"FID Score: {fid_score:.4f}\n")
        else:
            f.write("FID Score: Not calculated\n")
            
        if is_score is not None:
            f.write(f"Inception Score: {is_score:.4f}\n")
        else:
            f.write("Inception Score: Not calculated\n")
    
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main() 