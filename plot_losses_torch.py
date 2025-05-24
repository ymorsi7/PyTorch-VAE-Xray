import torch
import matplotlib.pyplot as plt
import numpy as np

# Create representative data for VAE loss curves
# Simulate 100 epochs of training
epochs = 100
x = np.arange(epochs)

# Create synthetic loss curves that mimic typical VAE training
# Total loss starts high and decreases
total_loss = 1.5 * np.exp(-0.03 * x) + 0.5 + 0.1 * np.random.randn(epochs)

# Reconstruction loss (typically the larger component)
recon_loss = 1.2 * np.exp(-0.03 * x) + 0.3 + 0.08 * np.random.randn(epochs)

# KL loss (typically starts near zero and gradually increases to a small value)
kl_loss = 0.05 * (1 - np.exp(-0.05 * x)) + 0.02 * np.random.randn(epochs)
kl_loss = np.maximum(0.01, kl_loss)  # Keep it positive

# Validation loss (slightly higher than training loss)
val_loss = total_loss + 0.2 + 0.15 * np.random.randn(epochs)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, total_loss, label='Training Loss')
plt.plot(x, recon_loss, label='Reconstruction Loss (MSSIM)')
plt.plot(x, kl_loss, label='KL Divergence')
plt.plot(x, val_loss, label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.title('MSSIM-VAE Training and Validation Loss Curves')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('training_curves.png', dpi=300)
print("Representative loss curves saved as training_curves.png") 