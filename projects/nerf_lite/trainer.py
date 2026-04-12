"""
Training loop for NeRF Lite.

This module implements the training pipeline for Neural Radiance Fields,
including ray batching, loss computation, and visualization callbacks.

Key Components:
    - TrainingConfig: Hyperparameters for training
    - NeRFTrainer: Main trainer class with training loop
    - Visualization and checkpointing utilities

Usage:
    python trainer.py --epochs 100 --synthetic
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_network import NeRFConfig, NeRFLiteMLP
from ray_utils import CameraIntrinsics, CameraPose, sample_points_along_rays
from volume_renderer import volume_render, compute_psnr
from dataset import SyntheticScene, SyntheticSceneDataset, RayBatchDataset


@dataclass
class TrainingConfig:
    """
    Training hyperparameters for NeRF Lite.

    Attributes:
        batch_size: Number of rays per training step
        num_samples: Number of samples per ray
        near: Near plane distance
        far: Far plane distance
        learning_rate: Initial learning rate
        weight_decay: AdamW weight decay
        num_epochs: Total training epochs
        lr_decay_step: Epochs between LR decay
        lr_decay_factor: LR multiplication factor at decay
        device: Training device ("cuda" or "cpu")
        save_dir: Directory for checkpoints
        save_every: Save checkpoint every N epochs
        log_every: Log metrics every N steps
        render_every: Render validation image every N epochs
        num_workers: DataLoader workers
    """
    # Data
    batch_size: int = 1024
    num_samples: int = 64
    near: float = 2.0
    far: float = 6.0

    # Optimization
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    num_epochs: int = 100
    lr_decay_step: int = 50
    lr_decay_factor: float = 0.5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpointing
    save_dir: str = "pretrained"
    save_every: int = 20

    # Logging
    log_every: int = 10
    render_every: int = 10

    # DataLoader
    num_workers: int = 0


@dataclass
class TrainingMetrics:
    """Container for training metrics history."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    psnr_values: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)


class NeRFTrainer:
    """
    Trainer for NeRF Lite models.

    Handles the full training pipeline including:
    - Ray batching and point sampling
    - Forward pass and volume rendering
    - Loss computation and backpropagation
    - Learning rate scheduling
    - Checkpointing and visualization

    Args:
        model: NeRFLiteMLP model
        config: TrainingConfig with hyperparameters
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
    """

    def __init__(
        self,
        model: NeRFLiteMLP,
        config: TrainingConfig,
        train_dataset: SyntheticSceneDataset,
        val_dataset: Optional[SyntheticSceneDataset] = None
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Create ray batch dataset for efficient training
        self.ray_dataset = RayBatchDataset(train_dataset, rays_per_sample=config.batch_size)

        # DataLoader
        self.train_loader = DataLoader(
            self.ray_dataset,
            batch_size=1,  # Already batched by RayBatchDataset
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.device == "cuda"
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_decay_step,
            gamma=config.lr_decay_factor
        )

        # Metrics
        self.metrics = TrainingMetrics()

        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Best model tracking
        self.best_psnr = 0.0

    def train_step(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        target_rgb: torch.Tensor
    ) -> Dict[str, float]:
        """
        Execute a single training step.

        Process:
        1. Sample points along rays
        2. Query network for density and color
        3. Volume render to get predicted RGB
        4. Compute MSE loss
        5. Backpropagate

        Args:
            ray_origins: [B, 3] ray start points
            ray_directions: [B, 3] ray directions
            target_rgb: [B, 3] ground truth colors

        Returns:
            Dictionary with loss values
        """
        self.model.train()
        device = self.config.device

        # Move to device
        ray_origins = ray_origins.to(device)
        ray_directions = ray_directions.to(device)
        target_rgb = target_rgb.to(device)

        # Sample points along rays
        points, z_vals = sample_points_along_rays(
            ray_origins, ray_directions,
            self.config.near, self.config.far,
            self.config.num_samples,
            stratified=True,
            device=device
        )

        # Flatten for network
        N_rays = points.shape[0]
        N_samples = points.shape[1]
        points_flat = points.reshape(-1, 3)

        # Expand directions to match samples
        directions_flat = ray_directions.unsqueeze(1).expand(-1, N_samples, -1)
        directions_flat = directions_flat.reshape(-1, 3)
        directions_flat = F.normalize(directions_flat, dim=-1)

        # Forward pass
        density, color = self.model(points_flat, directions_flat)

        # Reshape for volume rendering
        density = density.reshape(N_rays, N_samples, 1)
        color = color.reshape(N_rays, N_samples, 3)

        # Volume render
        rgb, depth, weights = volume_render(
            density, color, z_vals, ray_directions, white_background=True
        )

        # Compute loss
        loss = F.mse_loss(rgb, target_rgb)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return {"loss": loss.item()}

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
        for batch in pbar:
            # Squeeze batch dimension from DataLoader
            ray_origins = batch["ray_origins"].squeeze(0)
            ray_directions = batch["ray_directions"].squeeze(0)
            target_rgb = batch["target_rgb"].squeeze(0)

            metrics = self.train_step(ray_origins, ray_directions, target_rgb)
            total_loss += metrics["loss"]
            num_batches += 1

            if num_batches % self.config.log_every == 0:
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Run validation on the validation set.

        Returns:
            Tuple of (validation loss, PSNR)
        """
        if self.val_dataset is None:
            return 0.0, 0.0

        self.model.eval()
        device = self.config.device

        total_loss = 0.0
        total_psnr = 0.0

        for idx in range(len(self.val_dataset)):
            sample = self.val_dataset[idx]

            ray_origins = sample["ray_origins"].to(device)
            ray_directions = sample["ray_directions"].to(device)
            target_rgb = sample["target_rgb"].to(device)

            # Render in chunks to avoid OOM
            chunk_size = self.config.batch_size
            all_rgb = []

            for i in range(0, ray_origins.shape[0], chunk_size):
                chunk_origins = ray_origins[i:i + chunk_size]
                chunk_directions = ray_directions[i:i + chunk_size]

                # Sample points
                points, z_vals = sample_points_along_rays(
                    chunk_origins, chunk_directions,
                    self.config.near, self.config.far,
                    self.config.num_samples,
                    stratified=False,
                    device=device
                )

                # Forward pass
                N_rays = points.shape[0]
                N_samples = points.shape[1]
                points_flat = points.reshape(-1, 3)
                directions_flat = chunk_directions.unsqueeze(1).expand(-1, N_samples, -1)
                directions_flat = directions_flat.reshape(-1, 3)
                directions_flat = F.normalize(directions_flat, dim=-1)

                density, color = self.model(points_flat, directions_flat)
                density = density.reshape(N_rays, N_samples, 1)
                color = color.reshape(N_rays, N_samples, 3)

                rgb, _, _ = volume_render(
                    density, color, z_vals, chunk_directions, white_background=True
                )
                all_rgb.append(rgb)

            pred_rgb = torch.cat(all_rgb, dim=0)

            # Compute metrics
            loss = F.mse_loss(pred_rgb, target_rgb)
            psnr = compute_psnr(pred_rgb, target_rgb)

            total_loss += loss.item()
            total_psnr += psnr.item()

        num_views = len(self.val_dataset)
        return total_loss / num_views, total_psnr / num_views

    @torch.no_grad()
    def render_image(
        self,
        pose: CameraPose,
        intrinsics: CameraIntrinsics
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a complete image from a camera viewpoint.

        Args:
            pose: Camera pose
            intrinsics: Camera intrinsics

        Returns:
            Tuple of (RGB image, depth map)
        """
        from ray_utils import generate_rays

        self.model.eval()
        device = self.config.device
        H, W = intrinsics.height, intrinsics.width

        # Generate rays
        ray_origins, ray_directions = generate_rays(intrinsics, pose, device)

        all_rgb = []
        all_depth = []
        chunk_size = self.config.batch_size

        for i in range(0, ray_origins.shape[0], chunk_size):
            chunk_origins = ray_origins[i:i + chunk_size]
            chunk_directions = ray_directions[i:i + chunk_size]

            # Sample points
            points, z_vals = sample_points_along_rays(
                chunk_origins, chunk_directions,
                self.config.near, self.config.far,
                self.config.num_samples,
                stratified=False,
                device=device
            )

            # Forward pass
            N_rays = points.shape[0]
            N_samples = points.shape[1]
            points_flat = points.reshape(-1, 3)
            directions_flat = chunk_directions.unsqueeze(1).expand(-1, N_samples, -1)
            directions_flat = directions_flat.reshape(-1, 3)
            directions_flat = F.normalize(directions_flat, dim=-1)

            density, color = self.model(points_flat, directions_flat)
            density = density.reshape(N_rays, N_samples, 1)
            color = color.reshape(N_rays, N_samples, 3)

            rgb, depth, _ = volume_render(
                density, color, z_vals, chunk_directions, white_background=True
            )

            all_rgb.append(rgb)
            all_depth.append(depth)

        rgb = torch.cat(all_rgb, dim=0).reshape(H, W, 3)
        depth = torch.cat(all_depth, dim=0).reshape(H, W)

        return rgb, depth

    def save_visualization(self, epoch: int):
        """Save rendered images for visualization."""
        try:
            import matplotlib.pyplot as plt

            # Render from a validation view
            if self.val_dataset is not None:
                pose = self.val_dataset.get_pose(0)
                intrinsics = self.val_dataset.get_intrinsics()
                target = self.val_dataset.images[0]
            else:
                pose = self.train_dataset.get_pose(0)
                intrinsics = self.train_dataset.get_intrinsics()
                target = self.train_dataset.images[0]

            rgb, depth = self.render_image(pose, intrinsics)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(target.cpu().numpy())
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")

            axes[1].imshow(rgb.cpu().numpy())
            axes[1].set_title(f"Rendered (Epoch {epoch + 1})")
            axes[1].axis("off")

            axes[2].imshow(depth.cpu().numpy(), cmap="viridis")
            axes[2].set_title("Depth")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(self.save_dir / f"render_epoch_{epoch + 1:03d}.png", dpi=100)
            plt.close()

        except ImportError:
            pass  # Skip if matplotlib not available

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": self.metrics,
            "config": self.config
        }

        torch.save(checkpoint, self.save_dir / "nerf_latest.pth")

        if is_best:
            torch.save(checkpoint, self.save_dir / "nerf_best.pth")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.metrics = checkpoint["metrics"]
        return checkpoint["epoch"]

    def train(self) -> TrainingMetrics:
        """
        Full training loop.

        Returns:
            TrainingMetrics with training history
        """
        print(f"Training on {self.config.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training views: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Validation views: {len(self.val_dataset)}")
        print()

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            self.metrics.train_losses.append(train_loss)
            self.metrics.learning_rates.append(self.scheduler.get_last_lr()[0])

            # Validation
            val_loss, psnr = self.validate()
            self.metrics.val_losses.append(val_loss)
            self.metrics.psnr_values.append(psnr)

            # Update learning rate
            self.scheduler.step()

            # Logging
            print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | PSNR: {psnr:.2f} dB")

            # Save best model
            is_best = psnr > self.best_psnr
            if is_best:
                self.best_psnr = psnr

            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, is_best)

            # Visualization
            if (epoch + 1) % self.config.render_every == 0:
                self.save_visualization(epoch)

        # Final save
        self.save_checkpoint(self.config.num_epochs - 1, is_best=False)

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")

        return self.metrics


def train_synthetic(args):
    """Train on synthetic scene."""
    # Create scene
    scene_config = SyntheticScene(
        num_spheres=args.num_spheres,
        seed=args.seed
    )

    # Create datasets
    train_dataset = SyntheticSceneDataset(
        scene_config=scene_config,
        num_views=args.num_views,
        image_size=args.image_size,
        camera_distance=4.0,
        split="train",
        device="cpu"  # Load on CPU, move to GPU during training
    )

    val_dataset = SyntheticSceneDataset(
        scene_config=scene_config,
        num_views=max(10, args.num_views // 10),
        image_size=args.image_size,
        camera_distance=4.0,
        split="val",
        device="cpu"
    )

    # Create model
    model_config = NeRFConfig(
        encoding_type=args.encoding,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    model = NeRFLiteMLP(model_config)

    # Create trainer
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=args.save_dir,
        render_every=args.render_every
    )

    trainer = NeRFTrainer(model, train_config, train_dataset, val_dataset)

    # Train
    metrics = trainer.train()

    # Save training curves
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(metrics.train_losses, label="Train")
        axes[0].plot(metrics.val_losses, label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].set_yscale("log")

        axes[1].plot(metrics.psnr_values)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("PSNR (dB)")
        axes[1].set_title("Validation PSNR")

        plt.tight_layout()
        plt.savefig(Path(args.save_dir) / "training_curves.png", dpi=150)
        plt.close()
        print(f"Saved training curves to {args.save_dir}/training_curves.png")
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeRF Lite")

    # Dataset
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    parser.add_argument("--num_views", type=int, default=100, help="Number of training views")
    parser.add_argument("--image_size", type=int, default=64, help="Image resolution")
    parser.add_argument("--num_spheres", type=int, default=3, help="Number of spheres in scene")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model
    parser.add_argument("--encoding", type=str, default="hash", choices=["hash", "positional"])
    parser.add_argument("--hidden_dim", type=int, default=64, help="MLP hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of MLP layers")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Rays per batch")
    parser.add_argument("--num_samples", type=int, default=64, help="Samples per ray")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    # Output
    parser.add_argument("--save_dir", type=str, default="pretrained", help="Save directory")
    parser.add_argument("--render_every", type=int, default=10, help="Render every N epochs")

    args = parser.parse_args()

    if args.synthetic:
        train_synthetic(args)
    else:
        print("Only synthetic dataset is currently supported.")
        print("Use --synthetic flag to train on procedural scene.")
