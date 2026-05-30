"""
Training Module for Latent Diffusion Model
Author: Preethi Dommaraju
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Custom dataset for loading training images"""
    def __init__(self, data_path, image_size=256):
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.images = self._load_images()

    def _load_images(self):
        images = []
        if os.path.exists(self.data_path):
            for f in os.listdir(self.data_path):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    images.append(os.path.join(self.data_path, f))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        image = Image.open(self.images[idx]).convert('RGB')
        return self.transform(image)


class Trainer:
    """Training pipeline for Latent Diffusion Model"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config["device"]
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["learning_rate"]
        )
        self.criterion = nn.MSELoss()

    def train(self, data_path):
        """Train the LDM model"""
        dataset = ImageDataset(data_path, self.config["image_size"])
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True
        )

        logger.info(f"Training on {len(dataset)} images")
        logger.info(f"Epochs: {self.config['num_epochs']}")

        for epoch in range(self.config["num_epochs"]):
            total_loss = 0
            for batch_idx, images in enumerate(dataloader):
                images = images.to(self.device)
                t = torch.randint(1, 1000, (1,)).item()

                self.optimizer.zero_grad()
                reconstructed, noise = self.model(images, t)
                loss = self.criterion(reconstructed, images)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.config['num_epochs']}] Loss: {avg_loss:.4f}")
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint"""
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/ldm_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def evaluate(self, test_data_path):
        """Evaluate model on test data"""
        dataset = ImageDataset(test_data_path, self.config["image_size"])
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"])
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images in dataloader:
                images = images.to(self.device)
                t = torch.randint(1, 1000, (1,)).item()
                reconstructed, _ = self.model(images, t)
                loss = self.criterion(reconstructed, images)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss
