"""
Training script for Sketch-to-Image Diffusion model.

This module provides:
1. Trainer class with training loop
2. Checkpoint saving/loading
3. Progress visualization
4. CLI interface for training

Usage:
    python train.py --epochs 100 --batch-size 16
    python train.py --synthetic  # Quick test with synthetic data
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from neural_network import TinySketchUNet
from diffusion_utils import NoiseScheduler, ddim_sample
from dataset import SketchImageDataset, SyntheticSketchDataset, download_edges2shoes


class Trainer:
    """
    Trainer for sketch-to-image diffusion model.

    Handles training loop, checkpointing, and visualization.
    """

    def __init__(
        self,
        model: TinySketchUNet,
        scheduler: NoiseScheduler,
        lr: float = 1e-4,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device

        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.lr_scheduler = None  # Set during training

        # Training history
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for sketch, target in pbar:
            sketch = sketch.to(self.device)
            target = target.to(self.device)

            # Sample random timesteps
            batch_size = target.shape[0]
            t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)

            # Add noise to target images
            noise = torch.randn_like(target)
            noisy_target = self.scheduler.add_noise(target, noise, t)

            # Concatenate noisy image with sketch condition
            model_input = torch.cat([noisy_target, sketch], dim=1)

            # Predict noise
            noise_pred = self.model(model_input, t)

            # MSE loss between predicted and actual noise
            loss = F.mse_loss(noise_pred, noise)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for sketch, target in dataloader:
            sketch = sketch.to(self.device)
            target = target.to(self.device)

            batch_size = target.shape[0]
            t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)

            noise = torch.randn_like(target)
            noisy_target = self.scheduler.add_noise(target, noise, t)

            model_input = torch.cat([noisy_target, sketch], dim=1)
            noise_pred = self.model(model_input, t)

            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: int,
        save_dir: str = "pretrained",
        save_every: int = 10
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Set up learning rate scheduler
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print("-" * 50)

        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                val_str = f", Val Loss: {val_loss:.4f}"
            else:
                val_str = ""

            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}{val_str}")

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch}.pth")
                print(f"  Saved checkpoint at epoch {epoch}")

            # Save best model
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(save_path / "best_model.pth")

        # Save final model
        self.save_checkpoint(save_path / "sketch2image_final.pth")
        print(f"\nTraining complete! Final model saved to {save_path / 'sketch2image_final.pth'}")

        # Plot training history
        self.plot_history(save_path / "training_history.png")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        print(f"Loaded checkpoint from {path}")

    def plot_history(self, save_path: Path):
        """Plot and save training history."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training history saved to {save_path}")

    @torch.no_grad()
    def generate_samples(self, sketches: torch.Tensor, num_steps: int = 20) -> torch.Tensor:
        """Generate images from sketches."""
        self.model.eval()
        sketches = sketches.to(self.device)
        return ddim_sample(self.model, sketches, self.scheduler, num_steps, show_progress=False)


def main():
    parser = argparse.ArgumentParser(description="Train Sketch-to-Image Diffusion")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=32, help="Image size")
    parser.add_argument("--timesteps", type=int, default=200, help="Diffusion timesteps")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create datasets
    if args.synthetic:
        print("Using synthetic dataset for quick testing...")
        train_dataset = SyntheticSketchDataset(num_samples=1000, image_size=args.image_size)
        val_dataset = SyntheticSketchDataset(num_samples=100, image_size=args.image_size)
    else:
        # Download dataset if needed
        dataset_path = download_edges2shoes(args.data_dir)
        train_dataset = SketchImageDataset(str(dataset_path), split="train", image_size=args.image_size)
        val_dataset = SketchImageDataset(str(dataset_path), split="val", image_size=args.image_size, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model and scheduler
    model = TinySketchUNet()
    scheduler = NoiseScheduler(num_timesteps=args.timesteps, device=device)

    # Create trainer
    trainer = Trainer(model, scheduler, lr=args.lr, device=device)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # Train
    trainer.train(train_loader, val_loader, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
