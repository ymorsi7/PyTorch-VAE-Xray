# Chest X-ray Image Generation with MSSIM-VAE

This project uses a Variational Autoencoder (VAE) with Structural Similarity Index Measure (MSSIM) loss to generate realistic chest X-ray images based on the [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE) implementation.

## Dataset Structure

The code assumes the chest X-ray dataset is structured as follows:

```
chest_xray/
  - chest_xray/
    - test/
      - NORMAL/
      - PNEUMONIA/
    - train/
      - NORMAL/
      - PNEUMONIA/
    - val/
      - NORMAL/
      - PNEUMONIA/
```

## Requirements

Install all requirements from the original PyTorch-VAE repository:

```bash
pip install -r requirements.txt
```

Additional requirements for evaluation with PyTorch-Ignite:
```bash
pip install -r ignite_requirements.txt
```

## Using Virtual Environment (Recommended)

We provide shell scripts that automatically set up a virtual environment with all required packages:

### Training with Virtual Environment

To train the model in a virtual environment:

```bash
./train_with_venv.sh
```

This script will:
1. Create a Python virtual environment in the `venv` directory (if it doesn't exist)
2. Install all required packages
3. Run the training script
4. Deactivate the virtual environment when complete

### Evaluation with Virtual Environment

To evaluate the trained model in a virtual environment:

```bash
./run_with_venv.sh
```

This script will:
1. Create a Python virtual environment in the `venv` directory (if it doesn't exist)
2. Install all required packages
3. Run the evaluation script
4. Deactivate the virtual environment when complete

## Training the Model Manually

### Option 1: Using the Python script

To train the model using the custom Python script:

```bash
python chest_xray_vae.py
```

This will train the MSSIM-VAE model with the default parameters defined in the script.

### Option 2: Using the original training script with config file

To train the model using the original PyTorch-VAE training script with our custom config:

```bash
python run.py -c configs/chest_xray_mssim_vae.yaml
```

## Generating Images and Evaluating the Model Manually

After training, use the evaluation script to generate new chest X-ray images and calculate FID and Inception scores using PyTorch-Ignite:

```bash
python evaluate_vae.py --checkpoint logs/ChestXray_MSSIM_VAE/version_X/checkpoints/last.ckpt --output_dir generated_images
```

Replace `version_X` with the appropriate version number from your training run, and adjust other parameters as needed:

- `--checkpoint`: Path to the trained model checkpoint
- `--output_dir`: Directory to save generated images (default: 'generated_images')
- `--num_samples`: Number of samples to generate (default: 1000)
- `--device`: Device to use (default: 'cuda' if available, otherwise 'cpu')
- `--seed`: Random seed for reproducibility (default: 42)

### FID and Inception Score with PyTorch-Ignite

The evaluation script uses PyTorch-Ignite's implementation of FID (Fréchet Inception Distance) and Inception Score metrics to assess the quality of the generated images. These metrics are based on features extracted by a pre-trained Inception v3 model:

- **FID**: Measures the distance between feature distributions of real and generated images. Lower values indicate better quality and more similarity to the real data distribution.

- **Inception Score**: Measures both the quality and diversity of generated images. Higher values indicate better performance.

The evaluation process includes:
1. Resizing images to 299x299 (required by Inception v3)
2. Using PIL's BILINEAR interpolation for better quality
3. Running the metrics on batches of real and generated images

## Report Generation

For the assignment, include the following in your report:

1. **Hyperparameter settings**: Include the values from `configs/chest_xray_mssim_vae.yaml`
2. **Training loss curves**: Get these from TensorBoard logs at `logs/ChestXray_MSSIM_VAE/version_X`
3. **Inception score and FID score**: Get these from the output of the evaluation script
4. **Generated image examples**: Include images from the `generated_images` directory

## TensorBoard Visualization

To view training progress and results:

```bash
tensorboard --logdir logs/ChestXray_MSSIM_VAE
```

## Why MSSIM-VAE?

The MSSIM-VAE uses a structural similarity index measure loss function, which is particularly well-suited for medical images. Unlike pixel-wise loss functions, SSIM focuses on preserving structural information, which is critical for maintaining diagnostically relevant features in medical imaging. 