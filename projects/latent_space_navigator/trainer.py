"""
VAE training with beta warmup for stable latent space learning.

Features:
- Beta warmup to prevent posterior collapse
- Checkpoint saving
- Latent space visualization during training
- MNIST auto-download via torchvision
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import matplotlib.pyplot as plt
import numpy as np

from neural_network import VAE, vae_loss


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    latent_dim: int = 2
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 50
    beta_target: float = 4.0
    beta_warmup_epochs: int = 10
    device: str = "cpu"
    save_dir: str = "pretrained"


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_losses: List[float] = field(default_factory=list)
    recon_losses: List[float] = field(default_factory=list)
    kl_losses: List[float] = field(default_factory=list)
    betas: List[float] = field(default_factory=list)
    current_epoch: int = 0
    total_epochs: int = 0


def get_mnist_loaders(batch_size: int = 128):
    """
    Load MNIST dataset with normalization.

    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


class VAETrainer:
    """
    Trainer for VAE with beta warmup.

    Beta warmup gradually increases the KL weight to prevent
    the model from ignoring the latent space early in training.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device

        self.model = VAE(latent_dim=config.latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

        self.metrics = TrainingMetrics()
        self.metrics.total_epochs = config.num_epochs

        os.makedirs(config.save_dir, exist_ok=True)

    def get_beta(self, epoch: int) -> float:
        """Calculate beta value with warmup."""
        if epoch < self.config.beta_warmup_epochs:
            return self.config.beta_target * (epoch / self.config.beta_warmup_epochs)
        return self.config.beta_target

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        beta = self.get_beta(epoch)

        for images, _ in train_loader:
            images = images.to(self.device)

            self.optimizer.zero_grad()

            recon, mu, log_var = self.model(images)
            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, log_var, beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

        n_samples = len(train_loader.dataset)
        return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples

    @torch.no_grad()
    def visualize_reconstructions(
        self,
        test_loader: DataLoader,
        epoch: int,
        num_images: int = 8
    ):
        """Save reconstruction comparison."""
        self.model.eval()

        images, _ = next(iter(test_loader))
        images = images[:num_images].to(self.device)

        recon, _, _ = self.model(images)

        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3))

        for i in range(num_images):
            axes[0, i].imshow(images[i, 0].cpu(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            axes[1, i].imshow(recon[i, 0].cpu(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{self.config.save_dir}/recon_epoch_{epoch:03d}.png', dpi=100)
        plt.close()

    @torch.no_grad()
    def visualize_latent_space(
        self,
        test_loader: DataLoader,
        epoch: int,
        max_samples: int = 5000
    ):
        """Save latent space scatter plot colored by digit."""
        self.model.eval()

        all_z = []
        all_labels = []

        for images, labels in test_loader:
            if len(all_z) * self.config.batch_size >= max_samples:
                break

            images = images.to(self.device)
            mu, _ = self.model.encode(images)
            all_z.append(mu.cpu())
            all_labels.append(labels)

        z = torch.cat(all_z, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        fig, ax = plt.subplots(figsize=(8, 8))

        scatter = ax.scatter(
            z[:, 0], z[:, 1],
            c=labels, cmap='tab10',
            alpha=0.5, s=5
        )

        cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
        cbar.set_label('Digit')

        ax.set_xlabel('z[0]')
        ax.set_ylabel('z[1]')
        ax.set_title(f'Latent Space (Epoch {epoch})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.config.save_dir}/latent_epoch_{epoch:03d}.png', dpi=100)
        plt.close()

    def save_checkpoint(self, filename: str = "vae_mnist.pth"):
        """Save model checkpoint."""
        path = os.path.join(self.config.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, filename: str = "vae_mnist.pth"):
        """Load model checkpoint."""
        path = os.path.join(self.config.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint.get('metrics', TrainingMetrics())
        print(f"Loaded checkpoint from {path}")

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        callback: Optional[Callable[[TrainingMetrics], None]] = None
    ) -> TrainingMetrics:
        """
        Full training loop.

        Args:
            train_loader: Training data
            test_loader: Test data for visualization
            callback: Optional progress callback

        Returns:
            Training metrics
        """
        print(f"Training VAE for {self.config.num_epochs} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")

        for epoch in range(self.config.num_epochs):
            self.metrics.current_epoch = epoch + 1
            beta = self.get_beta(epoch)

            loss, recon, kl = self.train_epoch(train_loader, epoch)

            self.metrics.train_losses.append(loss)
            self.metrics.recon_losses.append(recon)
            self.metrics.kl_losses.append(kl)
            self.metrics.betas.append(beta)

            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                  f"Loss: {loss:.4f} | Recon: {recon:.4f} | "
                  f"KL: {kl:.4f} | Beta: {beta:.2f}")

            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.visualize_reconstructions(test_loader, epoch + 1)
                self.visualize_latent_space(test_loader, epoch + 1)

            if callback:
                callback(self.metrics)

        self.save_checkpoint()
        self.visualize_reconstructions(test_loader, self.config.num_epochs)
        self.visualize_latent_space(test_loader, self.config.num_epochs)

        return self.metrics


def main():
    """Train VAE on MNIST."""
    config = TrainingConfig(
        latent_dim=2,
        batch_size=128,
        learning_rate=1e-3,
        num_epochs=50,
        beta_target=4.0,
        beta_warmup_epochs=10,
        device="cpu"
    )

    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(config.batch_size)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    trainer = VAETrainer(config)
    metrics = trainer.train(train_loader, test_loader)

    print("\nTraining complete!")
    print(f"Final loss: {metrics.train_losses[-1]:.4f}")
    print(f"Model saved to: {config.save_dir}/vae_mnist.pth")


if __name__ == "__main__":
    main()
