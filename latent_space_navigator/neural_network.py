"""
Variational Autoencoder (VAE) for MNIST latent space exploration.

Architecture designed for 2D latent space to enable direct visualization
and interactive navigation. CPU-friendly with ~100K parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """
    Convolutional encoder that maps 28x28 images to latent distribution parameters.

    Architecture:
        [1, 28, 28] -> Conv layers -> [128, 4, 4] -> FC -> (mu, log_var)
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 4 * 4, 256)

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.flatten(x)
        x = F.relu(self.fc(x))

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var


class Decoder(nn.Module):
    """
    Convolutional decoder that maps latent vectors to 28x28 images.

    Architecture:
        z -> FC -> [128, 4, 4] -> ConvTranspose layers -> [1, 28, 28]
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 128 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))

        x = x.view(-1, 128, 4, 4)

        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))

        x = x[:, :, 2:30, 2:30]

        return x


class VAE(nn.Module):
    """
    Variational Autoencoder combining encoder and decoder.

    Features:
        - 2D latent space for visualization
        - Reparameterization trick for training
        - Separate encode/decode methods for inference
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, sample, decode."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def sample(self, num_samples: int = 1, device: str = "cpu") -> torch.Tensor:
        """Generate samples from the prior distribution."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss: reconstruction + beta * KL divergence.

    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Latent mean
        log_var: Latent log variance
        beta: Weight for KL divergence (beta-VAE)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    print("Testing VAE architecture...")

    vae = VAE(latent_dim=2)
    print(f"Total parameters: {vae.count_parameters():,}")

    x = torch.randn(4, 1, 28, 28).clamp(0, 1)

    mu, log_var = vae.encode(x)
    print(f"Encoder output: mu {mu.shape}, log_var {log_var.shape}")

    z = torch.randn(4, 2)
    recon = vae.decode(z)
    print(f"Decoder output: {recon.shape}")

    recon, mu, log_var = vae(x)
    print(f"VAE output: recon {recon.shape}")

    total, recon_l, kl_l = vae_loss(recon, x, mu, log_var)
    print(f"Loss: total={total.item():.2f}, recon={recon_l.item():.2f}, kl={kl_l.item():.2f}")

    samples = vae.sample(num_samples=8)
    print(f"Samples: {samples.shape}")

    print("All tests passed!")
