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
            transforms.Resize(self.patch_size),
            transforms.CenterCrop(self.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize(self.patch_size),
            transforms.CenterCrop(self.patch_size),
            transforms.ToTensor(),
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
            'latent_dim': 128,
            'hidden_dims': [32, 64, 128, 256, 512],
            'loss_type': 'mssim',
            'alpha': 0.0025,  # Weight for KLD loss
            'kernel_size': 4,
            'M_N': 0.005,  # Weight coefficient for KLD in the ELBO
        },
        'data_params': {
            'data_path': '.',  # Path to the dataset directory
            'train_batch_size': 64,
            'val_batch_size': 64,
            'patch_size': 64,
            'num_workers': 4,
        },
        'exp_params': {
            'manual_seed': 1234,
            'LR': 0.005,
            'weight_decay': 0.0,
            'scheduler_gamma': 0.95,
            'kld_weight': 0.00025,
            'min_delta': 0.01
        },
        'trainer_params': {
            'accelerator': 'auto',
            'devices': 1,
            'max_epochs': 100,
            'gradient_clip_val': 1.5
        },
        'logging_params': {
            'save_dir': 'logs',
            'name': 'ChestXray_MSSIMVAE'
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
                                 name=config['logging_params']['name'])
    
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
    
    # Initialize trainer
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(save_top_k=2, 
                           dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), 
                           monitor="val_loss",
                           save_last=True),
        ],
        **config['trainer_params']
    )
    
    # Create directories for samples and reconstructions
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    
    # Train the model, possibly resuming from checkpoint
    print(f"======= Training {config['model_params']['name']} on Chest X-ray dataset =======")
    trainer.fit(experiment, datamodule=data, ckpt_path=checkpoint_path)
    
    # After training, load the best model
    checkpoint_path = os.path.join(tb_logger.log_dir, "checkpoints/val_loss.ckpt")
    if os.path.exists(checkpoint_path):
        experiment = VAEExperiment.load_from_checkpoint(checkpoint_path)
        model = experiment.model
        
        # Generate samples
        samples = generate_samples(model, num_samples=16)
        
        # Save samples
        import torchvision
        img_grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
        torchvision.utils.save_image(img_grid, os.path.join(tb_logger.log_dir, "Samples/generated_samples.png"))
        
        print(f"Generated samples saved to {os.path.join(tb_logger.log_dir, 'Samples/generated_samples.png')}")

if __name__ == "__main__":
    main() 