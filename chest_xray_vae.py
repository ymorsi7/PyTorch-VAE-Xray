import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from models import vae_models
from experiment import VAEXperiment

class ChestXrayDataset(Dataset):
    """
    Chest X-ray dataset for pneumonia detection
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform=None):
        self.data_dir = Path(data_path) / "chest_xray" / split
        self.transform = transform
        self.img_paths = []
        self.labels = []
        
        # Get all image files from both NORMAL and PNEUMONIA classes
        normal_dir = self.data_dir / "NORMAL"
        pneumonia_dir = self.data_dir / "PNEUMONIA"
        
        if normal_dir.exists():
            for img_path in normal_dir.glob("*.jpeg"):
                self.img_paths.append(img_path)
                self.labels.append(0)  # 0 for NORMAL
                
            for img_path in normal_dir.glob("*.jpg"):
                self.img_paths.append(img_path)
                self.labels.append(0)  # 0 for NORMAL
        
        if pneumonia_dir.exists():
            for img_path in pneumonia_dir.glob("*.jpeg"):
                self.img_paths.append(img_path)
                self.labels.append(1)  # 1 for PNEUMONIA
                
            for img_path in pneumonia_dir.glob("*.jpg"):
                self.img_paths.append(img_path)
                self.labels.append(1)  # 1 for PNEUMONIA
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Return image only since VAE doesn't use labels for training
        # But we still return labels in case needed for analysis
        return img, self.labels[idx]

class ChestXrayDataModule(LightningDataModule):
    """
    PyTorch Lightning data module for chest X-ray dataset
    """
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        patch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        # Define transforms for the training and validation sets
        train_transforms = transforms.Compose([
            transforms.Resize(self.patch_size + 8),  # Resize slightly larger for cropping
            transforms.RandomCrop(self.patch_size),  # Random crop for more variation
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),  # Slight rotation for robustness
            transforms.RandomAutocontrast(p=0.3),  # Enhance contrast sometimes
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),  # Sharpen some images
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1] for better training
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize(self.patch_size),
            transforms.CenterCrop(self.patch_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Same normalization for consistency
        ])
        
        # Create the datasets
        self.train_dataset = ChestXrayDataset(
            self.data_dir,
            split='train',
            transform=train_transforms,
        )
        
        self.val_dataset = ChestXrayDataset(
            self.data_dir,
            split='val',
            transform=val_transforms,
        )
        
        self.test_dataset = ChestXrayDataset(
            self.data_dir,
            split='test',
            transform=val_transforms,
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

def create_config():
    """Create a config for MSSIM-VAE with chest X-ray dataset parameters"""
    config = {
        'model_params': {
            'name': 'MSSIMVAE',
            'in_channels': 3,
            'latent_dim': 512,  # Increased from 128 to 512 for more details
            'hidden_dims': [32, 64, 128, 256, 512],  # Keeping original architecture
            'loss_type': 'mssim',
            'alpha': 0.0015,  # Reduced from 0.0025 for less blur
            'kernel_size': 4,
            'M_N': 0.0025,  # Reduced from 0.005 to reduce the KL regularization strength
        },
        'data_params': {
            'data_path': '.',  # Path to the dataset directory
            'train_batch_size': 64,  # Back to original batch size
            'val_batch_size': 64,
            'patch_size': 64,  # Back to original 64x64 to avoid dimension mismatch
            'num_workers': 4,
        },
        'exp_params': {
            'manual_seed': 1234,
            'LR': 0.0025,  # Reduced from 0.005 for more stable training
            'weight_decay': 0.00001,  # Added slight weight decay for regularization
            'scheduler_gamma': 0.97,  # Slower decay (was 0.95)
            'kld_weight': 0.00015,  # Reduced from 0.00025 to allow more detailed reconstructions
            'min_delta': 0.01
        },
        'trainer_params': {
            'accelerator': 'auto',
            'devices': 1,
            'max_epochs': 50,  # Reduced epochs for initial testing
            'gradient_clip_val': 1.0,  # Reduced from 1.5 for stability
            'log_every_n_steps': 20,  # Log less frequently to save space
            'enable_checkpointing': True,
            'enable_progress_bar': True,
        },
        'logging_params': {
            'save_dir': 'logs',
            'name': 'ChestXray_DetailedMSSIMVAE',  # New name to avoid confusion
            'flush_logs_every_n_steps': 100,  # Flush logs less frequently
            'log_save_interval': 5,  # Save logs less frequently
        }
    }
    return config

def generate_samples(model, num_samples=10):
    """Generate new chest X-ray samples from the trained model"""
    model.eval()
    with torch.no_grad():
        # Sample from the latent space
        z = torch.randn(num_samples, model.latent_dim).to(model.device)
        # Generate images
        samples = model.decode(z)
        
    return samples

def main():
    # Create configuration
    config = create_config()
    
    # Set up logger
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                 name=config['logging_params']['name'],
                                 log_graph=False,  # Don't save model graph to save space
                                 default_hp_metric=False)  # Don't log hyperparameters to save space
    
    # Set seed for reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    
    # Initialize model and experiment
    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model, config['exp_params'])
    
    # Initialize data module
    data = ChestXrayDataModule(**config['data_params'], pin_memory=config['trainer_params']['devices'] > 0)
    data.setup()
    
    # Check if there are existing checkpoints to resume from
    import glob
    from pathlib import Path
    
    checkpoint_path = None
    checkpoints = list(Path('logs').glob(f"{config['logging_params']['name']}/**/checkpoints/last.ckpt"))
    if checkpoints:
        checkpoint_path = str(checkpoints[0])
        print(f"Resuming training from checkpoint: {checkpoint_path}")
    
    # Initialize trainer with fewer callbacks
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[
            ModelCheckpoint(save_top_k=1,  # Keep only the best model
                           dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), 
                           monitor="val_loss",
                           save_last=True),
        ],
        **config['trainer_params']
    )
    
    # Create directories for samples and reconstructions - only a few samples
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    
    # Train the model, possibly resuming from checkpoint
    print(f"======= Training {config['model_params']['name']} on Chest X-ray dataset =======")
    trainer.fit(experiment, datamodule=data, ckpt_path=checkpoint_path)

if __name__ == "__main__":
    main() 