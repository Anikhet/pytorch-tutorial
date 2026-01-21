"""
Neural Cellular Automata Model

Based on "Growing Neural Cellular Automata" (Mordvintsev et al., 2020)
https://distill.pub/2020/growing-ca/

Each cell has a state vector that evolves through learned local rules,
allowing patterns to self-organize and regenerate when damaged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PerceptionKernel(nn.Module):
    """
    Sobel-like filters to perceive local gradients.
    Each cell "sees" its neighbors through gradient information.
    """

    def __init__(self, channels: int = 16):
        super().__init__()
        self.channels = channels

        # Sobel filters for x and y gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.T
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

        # Stack filters: identity, sobel_x, sobel_y for each channel
        # Result: 3 * channels filters
        kernel = torch.stack([identity, sobel_x, sobel_y], dim=0)
        kernel = kernel.unsqueeze(1).repeat(channels, 1, 1, 1)  # [3*C, 1, 3, 3]
        kernel = kernel.reshape(channels * 3, 1, 3, 3)

        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply perception filters.

        Args:
            x: State tensor [B, C, H, W]

        Returns:
            Perceived state [B, C*3, H, W]
        """
        b, c, h, w = x.shape
        # Apply depthwise convolution with circular padding
        x_pad = F.pad(x, (1, 1, 1, 1), mode='circular')

        # Reshape for grouped convolution
        perceived = F.conv2d(x_pad, self.kernel, groups=c)
        return perceived


class UpdateNetwork(nn.Module):
    """
    Small MLP that computes state updates from perceived information.
    """

    def __init__(self, perception_channels: int = 48, hidden_channels: int = 128, state_channels: int = 16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(perception_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, state_channels, 1, bias=False),
        )

        # Initialize final layer to zero for stable training
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralCellularAutomata(nn.Module):
    """
    Neural Cellular Automata that learns to grow patterns.

    The model learns local update rules that allow:
    - Growing complex patterns from a single seed
    - Self-repair when damaged
    - Stable persistent patterns

    Args:
        state_channels: Number of channels per cell (RGBA + hidden)
        hidden_channels: Size of hidden layer in update network
        cell_fire_rate: Probability of cell update per step (stochastic updates)
    """

    def __init__(
        self,
        state_channels: int = 16,
        hidden_channels: int = 128,
        cell_fire_rate: float = 0.5,
    ):
        super().__init__()

        self.state_channels = state_channels
        self.cell_fire_rate = cell_fire_rate

        # 4 visible channels (RGBA), rest are hidden
        self.visible_channels = 4
        self.hidden_channels_count = state_channels - self.visible_channels

        # Perception: identity + sobel_x + sobel_y = 3x channels
        self.perception = PerceptionKernel(state_channels)
        perception_channels = state_channels * 3

        # Update network
        self.update = UpdateNetwork(perception_channels, hidden_channels, state_channels)

    def alive_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determine which cells are "alive" based on alpha channel.
        A cell is alive if any neighbor has alpha > 0.1
        """
        alpha = x[:, 3:4, :, :]  # Alpha channel
        # Max pool to check neighbors
        alive = F.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1
        return alive.float()

    def forward(self, x: torch.Tensor, steps: int = 1, return_history: bool = False) -> torch.Tensor:
        """
        Run the NCA for multiple steps.

        Args:
            x: Current state [B, C, H, W]
            steps: Number of update steps
            return_history: If True, return all intermediate states

        Returns:
            Final state or list of states if return_history=True
        """
        history = [x] if return_history else None

        for _ in range(steps):
            x = self.step(x)
            if return_history:
                history.append(x)

        return history if return_history else x

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Single NCA update step."""
        # Pre-life mask
        pre_life_mask = self.alive_mask(x)

        # Perceive neighborhood
        perceived = self.perception(x)

        # Compute update
        dx = self.update(perceived)

        # Stochastic cell update (training stability)
        if self.training:
            stochastic_mask = (torch.rand_like(x[:, :1, :, :]) < self.cell_fire_rate).float()
            dx = dx * stochastic_mask

        # Apply update
        x = x + dx

        # Post-life mask (cells die if isolated)
        post_life_mask = self.alive_mask(x)
        life_mask = pre_life_mask * post_life_mask

        x = x * life_mask

        return x

    def create_seed(self, height: int, width: int, batch_size: int = 1, device: str = "cuda") -> torch.Tensor:
        """
        Create initial seed state (single pixel in center).
        """
        x = torch.zeros(batch_size, self.state_channels, height, width, device=device)
        # Initialize center pixel with alpha=1, others random
        cx, cy = width // 2, height // 2
        x[:, 3, cy, cx] = 1.0  # Alpha = 1
        x[:, :3, cy, cx] = 0.0  # RGB = 0 (will be learned)
        return x

    def get_rgba(self, x: torch.Tensor) -> torch.Tensor:
        """Extract RGBA channels (first 4 channels)."""
        return x[:, :4, :, :]

    def get_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Extract RGB channels, alpha-premultiplied."""
        rgba = self.get_rgba(x)
        rgb = rgba[:, :3, :, :]
        alpha = rgba[:, 3:4, :, :]
        return torch.clamp(rgb * alpha, 0, 1)

    def damage(self, x: torch.Tensor, damage_type: str = "circle", **kwargs) -> torch.Tensor:
        """
        Apply damage to the state for regeneration demos.

        Args:
            x: Current state
            damage_type: "circle", "half", "random", "scratch"
        """
        b, c, h, w = x.shape
        mask = torch.ones_like(x)

        if damage_type == "circle":
            cx = kwargs.get("cx", w // 2)
            cy = kwargs.get("cy", h // 2)
            radius = kwargs.get("radius", min(h, w) // 4)

            y, xx = torch.meshgrid(
                torch.arange(h, device=x.device),
                torch.arange(w, device=x.device),
                indexing='ij'
            )
            dist = ((xx - cx) ** 2 + (y - cy) ** 2).float().sqrt()
            circle_mask = (dist > radius).float()
            mask = mask * circle_mask.unsqueeze(0).unsqueeze(0)

        elif damage_type == "half":
            direction = kwargs.get("direction", "right")
            if direction == "right":
                mask[:, :, :, w//2:] = 0
            elif direction == "left":
                mask[:, :, :, :w//2] = 0
            elif direction == "top":
                mask[:, :, :h//2, :] = 0
            else:
                mask[:, :, h//2:, :] = 0

        elif damage_type == "random":
            prob = kwargs.get("prob", 0.3)
            random_mask = (torch.rand(b, 1, h, w, device=x.device) > prob).float()
            mask = mask * random_mask

        elif damage_type == "scratch":
            x1, y1 = kwargs.get("start", (0, h // 2))
            x2, y2 = kwargs.get("end", (w, h // 2))
            thickness = kwargs.get("thickness", 5)

            y, xx = torch.meshgrid(
                torch.arange(h, device=x.device),
                torch.arange(w, device=x.device),
                indexing='ij'
            )
            # Line distance approximation
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dist = torch.abs((y - y1) * dx - (xx - x1) * dy) / length
                line_mask = (dist > thickness).float()
                mask = mask * line_mask.unsqueeze(0).unsqueeze(0)

        return x * mask


class SamplePool:
    """
    Pool of training samples for stable training.
    Helps the model learn to maintain stable patterns over time.
    """

    def __init__(self, size: int = 1024, state_shape: Tuple[int, ...] = (16, 64, 64)):
        self.size = size
        self.state_shape = state_shape
        self.pool = None

    def initialize(self, seed_fn, device: str = "cuda"):
        """Initialize pool with seeds."""
        c, h, w = self.state_shape
        self.pool = seed_fn(h, w, self.size, device)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, np.ndarray]:
        """Sample batch from pool."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = self.pool[indices].clone()
        return batch, indices

    def update(self, indices: np.ndarray, states: torch.Tensor, reset_prob: float = 0.0):
        """Update pool with new states, optionally resetting some to seed."""
        self.pool[indices] = states.detach()

        # Randomly reset some samples to seed
        if reset_prob > 0:
            reset_mask = np.random.random(len(indices)) < reset_prob
            if reset_mask.any():
                reset_indices = indices[reset_mask]
                c, h, w = self.state_shape
                device = self.pool.device
                seeds = torch.zeros(len(reset_indices), c, h, w, device=device)
                seeds[:, 3, h//2, w//2] = 1.0  # Alpha = 1 at center
                self.pool[reset_indices] = seeds
