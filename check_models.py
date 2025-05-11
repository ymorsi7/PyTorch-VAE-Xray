import sys
from models import vae_models

if __name__ == "__main__":
    print("Available VAE models:")
    for model_name in vae_models.keys():
        print(f"- {model_name}") 