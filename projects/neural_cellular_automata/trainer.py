"""
Training pipeline for Neural Cellular Automata.

Trains the NCA to grow target images from a seed, with:
- Pool-based sampling for stable persistent patterns
- Damage augmentation for regeneration ability
- Multi-step training through backprop-through-time
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Optional, Callable
import urllib.request
from tqdm import tqdm

from nca_model import NeuralCellularAutomata, SamplePool


def load_emoji(emoji: str = "🦎", size: int = 64, padding: int = 8) -> torch.Tensor:
    """
    Load emoji as target image using system fonts.

    Returns RGBA tensor [1, 4, H, W] normalized to [0, 1]
    """
    try:
        from PIL import ImageDraw, ImageFont

        # Create image with transparent background
        img_size = size - 2 * padding
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Try to find a font with emoji support
        font_paths = [
            "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux
            "C:\\Windows\\Fonts\\seguiemj.ttf",  # Windows
        ]

        font = None
        for path in font_paths:
            if Path(path).exists():
                try:
                    font = ImageFont.truetype(path, img_size)
                    break
                except Exception:
                    continue

        if font is None:
            # Fallback: download a simple test image
            return _download_test_target(size)

        # Draw emoji centered
        bbox = draw.textbbox((0, 0), emoji, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (size - text_w) // 2 - bbox[0]
        y = (size - text_h) // 2 - bbox[1]
        draw.text((x, y), emoji, font=font, embedded_color=True)

        # Convert to tensor
        img_array = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return tensor

    except Exception as e:
        print(f"Could not load emoji: {e}, using fallback")
        return _download_test_target(size)


def _download_test_target(size: int = 64) -> torch.Tensor:
    """Download a simple test target image."""
    # Create a simple colorful pattern as fallback
    img = torch.zeros(1, 4, size, size)

    # Create a lizard-like shape
    cx, cy = size // 2, size // 2

    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')

    # Body (ellipse)
    body_mask = ((x - cx) / 12) ** 2 + ((y - cy) / 8) ** 2 <= 1

    # Head
    head_mask = ((x - cx - 10) / 6) ** 2 + ((y - cy) / 5) ** 2 <= 1

    # Tail
    tail_mask = ((x - cx + 15) / 10) ** 2 + ((y - cy) / 3) ** 2 <= 1

    # Combine
    full_mask = body_mask | head_mask | tail_mask

    # Color: green lizard
    img[0, 0, full_mask] = 0.2  # R
    img[0, 1, full_mask] = 0.8  # G
    img[0, 2, full_mask] = 0.3  # B
    img[0, 3, full_mask] = 1.0  # A

    return img


def load_image(path: str, size: int = 64) -> torch.Tensor:
    """
    Load an image file as target.

    Returns RGBA tensor [1, 4, H, W] normalized to [0, 1]
    """
    img = Image.open(path).convert('RGBA')
    img = img.resize((size, size), Image.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def create_geometric_target(shape: str = "circle", size: int = 64) -> torch.Tensor:
    """Create geometric shape targets for quick testing."""
    img = torch.zeros(1, 4, size, size)
    cx, cy = size // 2, size // 2
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')

    if shape == "circle":
        mask = ((x - cx) ** 2 + (y - cy) ** 2) <= (size // 3) ** 2
        img[0, 0, mask] = 1.0  # Red
        img[0, 1, mask] = 0.3
        img[0, 2, mask] = 0.3

    elif shape == "square":
        margin = size // 4
        mask = (x >= margin) & (x < size - margin) & (y >= margin) & (y < size - margin)
        img[0, 0, mask] = 0.2
        img[0, 1, mask] = 0.5
        img[0, 2, mask] = 1.0  # Blue

    elif shape == "triangle":
        center = size // 2
        height = size // 2
        # Simple triangle approximation
        mask = (y > size // 4) & (y < size * 3 // 4)
        row_width = (y - size // 4) * 0.8
        mask = mask & (torch.abs(x - center) < row_width)
        img[0, 0, mask] = 0.1
        img[0, 1, mask] = 0.9  # Green
        img[0, 2, mask] = 0.3

    elif shape == "star":
        # 5-pointed star using polar coordinates
        angles = torch.atan2(y.float() - cy, x.float() - cx)
        radii = torch.sqrt((x.float() - cx) ** 2 + (y.float() - cy) ** 2)
        star_radius = size // 3 * (0.5 + 0.5 * torch.cos(5 * angles))
        mask = radii < star_radius
        img[0, 0, mask] = 1.0  # Yellow
        img[0, 1, mask] = 0.85
        img[0, 2, mask] = 0.1

    elif shape == "heart":
        # Heart shape
        xn = (x.float() - cx) / (size / 4)
        yn = -(y.float() - cy) / (size / 4) + 0.5
        heart = (xn ** 2 + yn ** 2 - 1) ** 3 - xn ** 2 * yn ** 3
        mask = heart <= 0
        img[0, 0, mask] = 0.9  # Pink/Red
        img[0, 1, mask] = 0.2
        img[0, 2, mask] = 0.4

    else:
        raise ValueError(f"Unknown shape: {shape}")

    img[0, 3, mask] = 1.0  # Alpha
    return img


class NCATrainer:
    """
    Trainer for Neural Cellular Automata.

    Features:
    - Pool-based sampling for stable patterns
    - Gradient checkpointing for memory efficiency
    - Damage augmentation for regeneration
    - Configurable loss functions
    """

    def __init__(
        self,
        model: NeuralCellularAutomata,
        target: torch.Tensor,
        device: str = "cuda",
        lr: float = 2e-3,
        pool_size: int = 1024,
        batch_size: int = 8,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

        # Target image (RGBA)
        self.target = target.to(device)
        _, _, self.height, self.width = target.shape

        # Sample pool for stable training
        state_shape = (model.state_channels, self.height, self.width)
        self.pool = SamplePool(size=pool_size, state_shape=state_shape)
        self.pool.initialize(model.create_seed, device)

        # Optimizer
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = None

        # Training history
        self.history = {
            "loss": [],
            "step": [],
        }

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between output RGBA and target RGBA.
        Uses MSE on RGB (alpha-weighted) + alpha loss.
        """
        # Extract RGBA
        out_rgba = self.model.get_rgba(output)
        target_rgba = target.expand(out_rgba.shape[0], -1, -1, -1)

        # Separate channels
        out_rgb = out_rgba[:, :3]
        out_alpha = out_rgba[:, 3:4]
        target_rgb = target_rgba[:, :3]
        target_alpha = target_rgba[:, 3:4]

        # RGB loss (only where target has alpha)
        rgb_loss = F.mse_loss(out_rgb * target_alpha, target_rgb * target_alpha)

        # Alpha loss
        alpha_loss = F.mse_loss(out_alpha, target_alpha)

        return rgb_loss + alpha_loss

    def train_step(self, steps_range: tuple = (64, 96), damage_prob: float = 0.0) -> float:
        """
        Single training step.

        Args:
            steps_range: Range of NCA steps to run (min, max)
            damage_prob: Probability of applying damage to samples
        """
        self.model.train()

        # Sample from pool
        batch, indices = self.pool.sample(self.batch_size)

        # Optionally apply damage for regeneration training
        if damage_prob > 0 and np.random.random() < damage_prob:
            damage_types = ["circle", "half", "random"]
            damage_type = np.random.choice(damage_types)
            batch = self.model.damage(batch, damage_type)

        # Random number of steps
        n_steps = np.random.randint(steps_range[0], steps_range[1])

        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(batch, steps=n_steps)

        # Compute loss
        loss = self.compute_loss(output, self.target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        # Update pool (reset worst samples occasionally)
        with torch.no_grad():
            # Find samples with highest loss
            batch_losses = []
            for i in range(self.batch_size):
                single_loss = self.compute_loss(output[i:i+1], self.target)
                batch_losses.append(single_loss.item())

            # Reset worst sample
            worst_idx = np.argmax(batch_losses)
            output_copy = output.detach().clone()
            output_copy[worst_idx] = self.model.create_seed(
                self.height, self.width, 1, self.device
            )[0]

            self.pool.update(indices, output_copy)

        return loss.item()

    def train(
        self,
        n_steps: int = 5000,
        steps_range: tuple = (64, 96),
        damage_start: int = 2000,
        damage_prob: float = 0.5,
        log_every: int = 100,
        save_every: int = 1000,
        save_dir: str = "checkpoints",
        callback: Optional[Callable] = None,
    ):
        """
        Full training loop.

        Args:
            n_steps: Total training steps
            steps_range: Range of NCA steps per training step
            damage_start: Step to start damage augmentation
            damage_prob: Probability of damage after damage_start
            log_every: Log frequency
            save_every: Checkpoint frequency
            save_dir: Directory for checkpoints
            callback: Optional callback(step, loss, model) for visualization
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        pbar = tqdm(range(n_steps), desc="Training NCA")

        for step in pbar:
            # Apply damage after warmup
            current_damage_prob = damage_prob if step >= damage_start else 0.0

            loss = self.train_step(steps_range, current_damage_prob)

            self.history["loss"].append(loss)
            self.history["step"].append(step)

            if step % log_every == 0:
                pbar.set_postfix(loss=f"{loss:.4f}")

            if callback is not None and step % log_every == 0:
                callback(step, loss, self.model)

            if save_every > 0 and (step + 1) % save_every == 0:
                self.save_checkpoint(save_path / f"nca_step_{step+1}.pth")

        # Final save
        self.save_checkpoint(save_path / "nca_final.pth")
        print(f"Training complete! Model saved to {save_path}")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {"loss": [], "step": []})


def train_quick_demo(
    target_type: str = "circle",
    size: int = 64,
    n_steps: int = 2000,
    device: str = "cuda",
):
    """Quick training demo with geometric shapes."""
    print(f"Training NCA on {target_type} target...")

    # Create target
    if target_type in ["circle", "square", "triangle", "star", "heart"]:
        target = create_geometric_target(target_type, size)
    elif target_type == "emoji":
        target = load_emoji("🦎", size)
    else:
        target = create_geometric_target("circle", size)

    # Create model
    model = NeuralCellularAutomata(
        state_channels=16,
        hidden_channels=128,
        cell_fire_rate=0.5,
    )

    # Create trainer
    trainer = NCATrainer(
        model=model,
        target=target,
        device=device,
        lr=2e-3,
        pool_size=256,
        batch_size=8,
    )

    # Train
    trainer.train(
        n_steps=n_steps,
        steps_range=(64, 96),
        damage_start=n_steps // 2,
        damage_prob=0.3,
        log_every=100,
        save_every=500,
        save_dir="checkpoints",
    )

    return model, trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Neural Cellular Automata")
    parser.add_argument("--target", type=str, default="heart", choices=["circle", "square", "triangle", "star", "heart", "emoji"])
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    model, trainer = train_quick_demo(
        target_type=args.target,
        size=args.size,
        n_steps=args.steps,
        device=args.device,
    )
