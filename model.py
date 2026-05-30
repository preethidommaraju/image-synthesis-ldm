"""
Latent Diffusion Model Architecture
Author: Preethi Dommaraju
"""

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """VAE Encoder - compresses images to latent space"""
    def __init__(self, image_size, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * (image_size // 8) ** 2, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """VAE Decoder - reconstructs images from latent space"""
    def __init__(self, image_size, latent_dim):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.fc = nn.Linear(latent_dim, 256 * (image_size // 8) ** 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, self.image_size // 8, self.image_size // 8)
        return self.decoder(x)


class DiffusionProcess(nn.Module):
    """Diffusion Process - forward and reverse diffusion"""
    def __init__(self, latent_dim, num_timesteps=1000):
        super(DiffusionProcess, self).__init__()
        self.num_timesteps = num_timesteps
        self.latent_dim = latent_dim
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward_diffusion(self, z, t):
        """Add noise to latent representation"""
        noise = torch.randn_like(z)
        alpha = 1 - (t / self.num_timesteps)
        noisy_z = alpha * z + (1 - alpha) * noise
        return noisy_z, noise

    def reverse_diffusion(self, noisy_z, t):
        """Remove noise from latent representation"""
        t_tensor = torch.tensor([t / self.num_timesteps]).expand(noisy_z.size(0), 1)
        input_tensor = torch.cat([noisy_z, t_tensor], dim=1)
        predicted_noise = self.noise_predictor(input_tensor)
        return noisy_z - predicted_noise


class LatentDiffusionModel(nn.Module):
    """Complete Latent Diffusion Model"""
    def __init__(self, image_size=256, latent_dim=512):
        super(LatentDiffusionModel, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.encoder = Encoder(image_size, latent_dim)
        self.decoder = Decoder(image_size, latent_dim)
        self.diffusion = DiffusionProcess(latent_dim)
        logger.info(f"LDM initialized | Image: {image_size}x{image_size} | Latent dim: {latent_dim}")

    def forward(self, x, t):
        z = self.encoder(x)
        noisy_z, noise = self.diffusion.forward_diffusion(z, t)
        denoised_z = self.diffusion.reverse_diffusion(noisy_z, t)
        reconstructed = self.decoder(denoised_z)
        return reconstructed, noise

    def generate(self, num_samples=4, output_path="output"):
        """Generate high-resolution images"""
        import os
        os.makedirs(output_path, exist_ok=True)
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            for t in range(self.diffusion.num_timesteps, 0, -1):
                z = self.diffusion.reverse_diffusion(z, t)
            images = self.decoder(z)
        logger.info(f"Generated {num_samples} images saved to {output_path}")
        return images
