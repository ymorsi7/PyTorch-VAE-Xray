import os
import torch
import numpy as np
from pathlib import Path
import argparse
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import PIL.Image as Image
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

def save_generated_images(images, output_dir, prefix='generated'):
    """Save generated images to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a grid of sample images (for visualization)
    grid_img = vutils.make_grid(images[:64], nrow=8, normalize=True, padding=2)
    vutils.save_image(grid_img, os.path.join(output_dir, f"{prefix}_grid.png"))
    
    # Save individual images
    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        vutils.save_image(img, img_path, normalize=True)
    
    return os.path.join(output_dir, f"{prefix}_grid.png")

def evaluate_with_ignite_metrics(real_dataset, generated_images, device='cuda'):
    """Evaluate FID and Inception Score using PyTorch-Ignite metrics"""
    # Create data loaders
    real_loader = DataLoader(real_dataset, batch_size=64, shuffle=False)
    
    # Store generated images for batch processing
    generated_batches = []
    for i in range(0, len(generated_images), 64):
        end = min(i + 64, len(generated_images))
        generated_batches.append(generated_images[i:end])
    
    # Define evaluation step
    def evaluation_step(engine, batch):
        if isinstance(batch, list):  # Generated batch
            fake = interpolate(batch)
            return fake, None  # Only used for Inception Score
        else:  # Real batch from dataset
            real = interpolate(batch[0])
            return None, real  # Only used for FID
    
    # Create evaluator engine
    evaluator = Engine(evaluation_step)
    
    fid_score = None
    is_score = None
    
    # Try to calculate FID score
    try:
        fid_metric = FID(device=device)
        fid_metric.attach(evaluator, "fid")
        
        # Run evaluator on real data to calculate FID stats
        evaluator.run(real_loader)
        
        # Run evaluator on generated data
        evaluator.run(generated_batches)
        
        # Extract metrics
        metrics = evaluator.state.metrics
        fid_score = metrics.get('fid')
        
        print(f"FID Score: {fid_score:.4f}")
    except Exception as e:
        print(f"Error calculating FID score: {str(e)}")
        print("Continuing with other metrics...")
    
    # Try to calculate Inception Score
    try:
        # Create new evaluator for Inception Score
        is_evaluator = Engine(evaluation_step)
        is_metric = InceptionScore(device=device, output_transform=lambda x: x[0] if x[0] is not None else x[1])
        is_metric.attach(is_evaluator, "is")
        
        # Run evaluator on generated data
        is_evaluator.run(generated_batches)
        
        # Extract metrics
        metrics = is_evaluator.state.metrics
        is_score = metrics.get('is')
        
        print(f"Inception Score: {is_score:.4f}")
    except Exception as e:
        print(f"Error calculating Inception Score: {str(e)}")
    
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