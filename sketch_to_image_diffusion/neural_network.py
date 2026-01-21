"""
TinySketchUNet - A CPU-optimized U-Net for conditional diffusion.

This network takes a noisy image concatenated with a sketch condition
and predicts the noise to be removed. Designed for 32x32 images with
~150K parameters for fast CPU inference.

Architecture:
    Input: [batch, 6, 32, 32] (3 RGB noisy + 3 sketch)
    Output: [batch, 3, 32, 32] (predicted noise)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embeddings for timestep encoding.

    Transforms scalar timestep into a rich vector representation
    using sin/cos at different frequencies (from Transformer paper).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2

        # Compute frequency bands
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Apply to timesteps
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm and time embedding injection.

    Structure: Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU + Skip
    Time embedding is added after the first normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Skip connection (adjust channels if needed)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)

        # Add time embedding (broadcast to spatial dims)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.activation(h)

        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        # Skip connection
        return h + self.skip(x)


class DownBlock(nn.Module):
    """Downsampling block: ResidualBlock + MaxPool"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res_block(x, t_emb)
        return self.downsample(h), h  # Return downsampled and skip connection


class UpBlock(nn.Module):
    """Upsampling block: Upsample + Concat skip + ResidualBlock"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # Concatenate upsampled tensor with skip connection
        self.res_block = ResidualBlock(in_channels + skip_channels, out_channels, time_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.res_block(x, t_emb)


class TinySketchUNet(nn.Module):
    """
    Tiny U-Net for sketch-conditioned diffusion.

    Optimized for CPU inference at 32x32 resolution.
    Takes concatenated [noisy_image, sketch] as input.

    Args:
        in_channels: Input channels (6 = 3 noisy + 3 sketch)
        out_channels: Output channels (3 = predicted noise)
        time_emb_dim: Dimension of time embeddings
        base_channels: Base channel count (scales up in deeper layers)
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        time_emb_dim: int = 128,
        base_channels: int = 32
    ):
        super().__init__()

        # Time embedding network
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder (downsampling path)
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)      # 32 -> 16
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)  # 16 -> 8

        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Decoder (upsampling path)
        # up1: input 128, skip2 has 128 channels, output 64
        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_emb_dim)  # 8 -> 16
        # up2: input 64, skip1 has 64 channels, output 32
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_emb_dim)      # 16 -> 32

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, 6, 32, 32] (noisy image + sketch)
            t: Timestep tensor [batch]

        Returns:
            Predicted noise [batch, 3, 32, 32]
        """
        # Time embedding
        t_emb = self.time_embed(t.float())

        # Initial conv
        h = self.init_conv(x)

        # Encoder with skip connections
        h, skip1 = self.down1(h, t_emb)
        h, skip2 = self.down2(h, t_emb)

        # Bottleneck
        h = self.bottleneck(h, t_emb)

        # Decoder with skip connections
        h = self.up1(h, skip2, t_emb)
        h = self.up2(h, skip1, t_emb)

        # Final conv
        return self.final_conv(h)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test the network when run directly
if __name__ == "__main__":
    print("Testing TinySketchUNet...")

    # Create model
    model = TinySketchUNet()
    print(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 6, 32, 32)  # noisy + sketch
    t = torch.randint(0, 200, (batch_size,))

    with torch.no_grad():
        output = model(x, t)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")
