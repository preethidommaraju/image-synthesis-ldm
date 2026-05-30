"""
High-Resolution Image Synthesis with Latent Diffusion Models
Author: Preethi Dommaraju
"""

from model import LatentDiffusionModel
from train import Trainer
import torch

def main():
    print("High-Resolution Image Synthesis with Latent Diffusion Models")
    print("=" * 60)

    # Configuration
    config = {
        "image_size": 256,
        "latent_dim": 512,
        "num_epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    print(f"Using device: {config['device']}")
    print(f"Image size: {config['image_size']}x{config['image_size']}")
    print(f"Latent dimensions: {config['latent_dim']}")

    # Initialize model
    print("\n[1/3] Initializing Latent Diffusion Model...")
    model = LatentDiffusionModel(
        image_size=config["image_size"],
        latent_dim=config["latent_dim"]
    )
    print("✅ Model initialized successfully!")

    # Train model
    print("\n[2/3] Starting model training...")
    trainer = Trainer(model=model, config=config)
    trainer.train(data_path="data/training_images")
    print("✅ Model training completed!")

    # Generate images
    print("\n[3/3] Generating high-resolution images...")
    model.generate(num_samples=4, output_path="output/generated_images")
    print("✅ Images generated successfully!")

    print("=" * 60)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
