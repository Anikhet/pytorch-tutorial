"""
Neural Network components for NeRF Lite.

This module implements the core neural network architecture for Neural Radiance
Fields with both classic positional encoding and instant-NGP style hash encoding.

Key Components:
    - PositionalEncoding: Fourier features for coordinate encoding
    - HashEncoding: Multi-resolution learnable hash tables (instant-NGP)
    - NeRFLiteMLP: Lightweight MLP for density and color prediction

Mathematical Background:
    NeRF represents a scene as F: (x, d) -> (c, sigma)
    where x is 3D position, d is viewing direction,
    c is RGB color, and sigma is volume density.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NeRFConfig:
    """Configuration for NeRF Lite model.

    Attributes:
        encoding_type: Type of positional encoding ("positional" or "hash")
        num_frequencies_pos: Number of frequency bands for position encoding
        num_frequencies_dir: Number of frequency bands for direction encoding
        hash_table_size: Size of each hash table (power of 2 recommended)
        hash_num_levels: Number of multi-resolution levels
        hash_features_per_level: Features stored per hash table entry
        hash_base_resolution: Coarsest grid resolution
        hash_max_resolution: Finest grid resolution
        hidden_dim: Width of MLP hidden layers
        num_layers: Number of layers in density network
        num_layers_color: Number of layers in color network
        geo_feat_dim: Dimension of geometry features passed to color network
        density_activation: Activation for density output
    """
    # Encoding settings
    encoding_type: str = "hash"
    num_frequencies_pos: int = 6
    num_frequencies_dir: int = 4

    # Hash encoding (instant-NGP style)
    hash_table_size: int = 2**16
    hash_num_levels: int = 8
    hash_features_per_level: int = 2
    hash_base_resolution: int = 16
    hash_max_resolution: int = 128

    # Network architecture
    hidden_dim: int = 64
    num_layers: int = 4
    num_layers_color: int = 2
    geo_feat_dim: int = 15

    # Output
    density_activation: str = "softplus"


class PositionalEncoding(nn.Module):
    """
    Fourier feature encoding from the original NeRF paper.

    Maps low-dimensional input to high-dimensional space using sinusoidal
    functions at exponentially increasing frequencies:

        gamma(p) = (p, sin(2^0 * pi * p), cos(2^0 * pi * p),
                      sin(2^1 * pi * p), cos(2^1 * pi * p),
                      ...,
                      sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p))

    This encoding allows the MLP to learn high-frequency functions,
    overcoming the spectral bias of neural networks toward low frequencies.

    Reference: Mildenhall et al., "NeRF: Representing Scenes as Neural
    Radiance Fields for View Synthesis", ECCV 2020

    Args:
        input_dim: Dimension of input coordinates (3 for position, 3 for direction)
        num_frequencies: Number of frequency bands (L in the paper)
        include_input: Whether to include the original input in the output
    """

    def __init__(
        self,
        input_dim: int,
        num_frequencies: int = 10,
        include_input: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Compute output dimension
        self.output_dim = 2 * input_dim * num_frequencies
        if include_input:
            self.output_dim += input_dim

        # Precompute frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freq_bands = torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", 2.0 ** freq_bands * np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input coordinates.

        Args:
            x: Input tensor of shape [..., input_dim]

        Returns:
            Encoded tensor of shape [..., output_dim]
        """
        # Expand x for broadcasting: [..., input_dim, 1]
        x_expanded = x.unsqueeze(-1)

        # Multiply by frequencies: [..., input_dim, num_frequencies]
        scaled = x_expanded * self.freq_bands

        # Apply sin and cos: [..., input_dim, num_frequencies * 2]
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

        # Flatten: [..., input_dim * num_frequencies * 2]
        encoded = encoded.flatten(start_dim=-2)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded


class HashEncoding(nn.Module):
    """
    Multi-resolution hash encoding from instant-NGP.

    Instead of fixed sinusoidal encoding, this uses learnable feature vectors
    stored in hash tables at multiple resolutions. Each 3D position is:
    1. Quantized to vertices of grids at L different resolutions
    2. Hashed to indices in the corresponding hash tables
    3. Features are retrieved and trilinearly interpolated
    4. Concatenated across all L levels

    Key advantages over positional encoding:
    - 10-100x faster convergence
    - Learnable features adapt to scene complexity
    - Hash collisions are resolved by gradient averaging

    Reference: Müller et al., "Instant Neural Graphics Primitives with a
    Multiresolution Hash Encoding", SIGGRAPH 2022

    Args:
        table_size: Size of each hash table (T in the paper)
        num_levels: Number of resolution levels (L)
        features_per_level: Features per hash entry (F)
        base_resolution: Coarsest grid resolution (N_min)
        max_resolution: Finest grid resolution (N_max)
        bounding_box: Scene bounding box for normalization
    """

    def __init__(
        self,
        table_size: int = 2**16,
        num_levels: int = 8,
        features_per_level: int = 2,
        base_resolution: int = 16,
        max_resolution: int = 128,
        bounding_box: Tuple[float, float] = (-1.0, 1.0)
    ):
        super().__init__()
        self.table_size = table_size
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.bounding_box = bounding_box

        self.output_dim = num_levels * features_per_level

        # Compute resolution at each level (geometric progression)
        # N_l = floor(N_min * b^l) where b = exp(ln(N_max/N_min) / (L-1))
        if num_levels > 1:
            b = np.exp(np.log(max_resolution / base_resolution) / (num_levels - 1))
        else:
            b = 1.0

        resolutions = [int(np.floor(base_resolution * (b ** level)))
                       for level in range(num_levels)]
        self.register_buffer("resolutions", torch.tensor(resolutions))

        # Initialize hash tables as learnable embeddings
        # Each level has its own table of size T x F
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(table_size, features_per_level) * 0.01)
            for _ in range(num_levels)
        ])

        # Prime numbers for spatial hashing
        self.register_buffer(
            "primes",
            torch.tensor([1, 2654435761, 805459861], dtype=torch.long)
        )

    def _hash_coords(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        """
        Hash 3D integer coordinates to table indices.

        Uses spatial hashing: h(x,y,z) = (x*p1 XOR y*p2 XOR z*p3) mod T

        Args:
            coords: Integer coordinates [N, 3]
            level: Resolution level index

        Returns:
            Hash indices [N]
        """
        # XOR hashing with prime numbers
        x = coords[..., 0].long() * self.primes[0]
        y = coords[..., 1].long() * self.primes[1]
        z = coords[..., 2].long() * self.primes[2]
        hashed = (x ^ y ^ z) % self.table_size
        return hashed.abs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute hash encoding for 3D positions.

        Args:
            x: Positions in [-1, 1]^3, shape [N, 3]

        Returns:
            Encoded features, shape [N, num_levels * features_per_level]
        """
        # Normalize to [0, 1]
        bb_min, bb_max = self.bounding_box
        x_normalized = (x - bb_min) / (bb_max - bb_min)
        x_normalized = torch.clamp(x_normalized, 0.0, 1.0 - 1e-6)

        features = []

        for level in range(self.num_levels):
            resolution = self.resolutions[level].item()

            # Scale to grid coordinates
            x_scaled = x_normalized * resolution

            # Get corner coordinates (floor)
            x_floor = torch.floor(x_scaled).long()

            # Compute interpolation weights
            weights = x_scaled - x_floor.float()

            # Get all 8 corners of the voxel
            corner_offsets = torch.tensor([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
            ], device=x.device, dtype=torch.long)

            # Trilinear interpolation
            interpolated = torch.zeros(x.shape[0], self.features_per_level, device=x.device)

            for corner_idx, offset in enumerate(corner_offsets):
                corner = x_floor + offset

                # Compute weight for this corner
                w_x = (1 - offset[0]) * (1 - weights[:, 0]) + offset[0] * weights[:, 0]
                w_y = (1 - offset[1]) * (1 - weights[:, 1]) + offset[1] * weights[:, 1]
                w_z = (1 - offset[2]) * (1 - weights[:, 2]) + offset[2] * weights[:, 2]
                w = w_x * w_y * w_z

                # Hash corner and lookup features
                indices = self._hash_coords(corner, level)
                corner_features = self.hash_tables[level][indices]

                interpolated = interpolated + w.unsqueeze(-1) * corner_features

            features.append(interpolated)

        return torch.cat(features, dim=-1)


class NeRFLiteMLP(nn.Module):
    """
    Lightweight NeRF MLP with view-dependent color prediction.

    Architecture overview:

        Position Encoding ─────┐
                               ├─► [Density Network] ─► density (sigma)
                               │                    └─► geometry features
                               │                              │
        Direction Encoding ────┴──────────────────────────────┤
                                                              v
                                                  [Color Network] ─► RGB

    The density network is position-only (view-independent density),
    while the color network receives both geometry features and viewing
    direction to model view-dependent effects like specular reflections.

    This "Lite" version uses:
    - 4 layers instead of 8 (original NeRF)
    - 64 hidden units instead of 256
    - Optional hash encoding for 10x faster training

    Args:
        config: NeRFConfig with model hyperparameters
    """

    def __init__(self, config: Optional[NeRFConfig] = None):
        super().__init__()
        self.config = config or NeRFConfig()

        # Initialize position encoding
        if self.config.encoding_type == "hash":
            self.pos_encoding = HashEncoding(
                table_size=self.config.hash_table_size,
                num_levels=self.config.hash_num_levels,
                features_per_level=self.config.hash_features_per_level,
                base_resolution=self.config.hash_base_resolution,
                max_resolution=self.config.hash_max_resolution,
            )
            pos_input_dim = self.pos_encoding.output_dim
        else:
            self.pos_encoding = PositionalEncoding(
                input_dim=3,
                num_frequencies=self.config.num_frequencies_pos,
            )
            pos_input_dim = self.pos_encoding.output_dim

        # Direction encoding (always positional for simplicity)
        self.dir_encoding = PositionalEncoding(
            input_dim=3,
            num_frequencies=self.config.num_frequencies_dir,
        )
        dir_input_dim = self.dir_encoding.output_dim

        # Density network: position -> density + geometry features
        hidden_dim = self.config.hidden_dim
        density_layers = []

        in_dim = pos_input_dim
        for i in range(self.config.num_layers):
            density_layers.append(nn.Linear(in_dim, hidden_dim))
            density_layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        self.density_net = nn.Sequential(*density_layers)

        # Output heads for density network
        self.density_head = nn.Linear(hidden_dim, 1)  # sigma
        self.geo_feat_head = nn.Linear(hidden_dim, self.config.geo_feat_dim)

        # Color network: geometry features + direction -> RGB
        color_layers = []
        color_in_dim = self.config.geo_feat_dim + dir_input_dim

        for i in range(self.config.num_layers_color):
            color_layers.append(nn.Linear(color_in_dim, hidden_dim))
            color_layers.append(nn.ReLU(inplace=True))
            color_in_dim = hidden_dim

        self.color_net = nn.Sequential(*color_layers)
        self.color_head = nn.Linear(hidden_dim, 3)  # RGB

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict density and color for 3D points viewed from given directions.

        Args:
            positions: 3D positions, shape [N, 3] in [-1, 1]^3
            directions: Viewing directions, shape [N, 3] (normalized)

        Returns:
            density: Volume density (sigma), shape [N, 1]
            color: RGB color in [0, 1], shape [N, 3]
        """
        # Encode positions
        pos_encoded = self.pos_encoding(positions)

        # Density network forward pass
        h = self.density_net(pos_encoded)

        # Get density (with activation to ensure non-negative)
        raw_density = self.density_head(h)
        if self.config.density_activation == "softplus":
            density = F.softplus(raw_density - 1.0)  # Shift for stability
        else:
            density = F.relu(raw_density)

        # Get geometry features for color prediction
        geo_features = self.geo_feat_head(h)

        # Encode viewing direction
        dir_encoded = self.dir_encoding(directions)

        # Color network forward pass
        color_input = torch.cat([geo_features, dir_encoded], dim=-1)
        h_color = self.color_net(color_input)
        color = torch.sigmoid(self.color_head(h_color))  # RGB in [0, 1]

        return density, color

    def get_density(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get density only (faster for occupancy queries).

        Args:
            positions: 3D positions, shape [N, 3]

        Returns:
            density: Volume density, shape [N, 1]
        """
        pos_encoded = self.pos_encoding(positions)
        h = self.density_net(pos_encoded)
        raw_density = self.density_head(h)

        if self.config.density_activation == "softplus":
            return F.softplus(raw_density - 1.0)
        return F.relu(raw_density)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test of the components
    print("Testing NeRF Lite components...")

    # Test positional encoding
    pos_enc = PositionalEncoding(input_dim=3, num_frequencies=6)
    x = torch.randn(10, 3)
    encoded = pos_enc(x)
    print(f"Positional encoding: {x.shape} -> {encoded.shape}")
    assert encoded.shape == (10, pos_enc.output_dim)

    # Test hash encoding
    hash_enc = HashEncoding(num_levels=4, features_per_level=2)
    x = torch.rand(10, 3) * 2 - 1  # [-1, 1]
    encoded = hash_enc(x)
    print(f"Hash encoding: {x.shape} -> {encoded.shape}")
    assert encoded.shape == (10, hash_enc.output_dim)

    # Test full model
    config = NeRFConfig(encoding_type="hash", hidden_dim=64)
    model = NeRFLiteMLP(config)
    print(f"Model parameters: {model.count_parameters():,}")

    positions = torch.rand(100, 3) * 2 - 1
    directions = F.normalize(torch.randn(100, 3), dim=-1)

    density, color = model(positions, directions)
    print(f"Forward pass: positions {positions.shape} -> density {density.shape}, color {color.shape}")
    assert density.shape == (100, 1)
    assert color.shape == (100, 3)
    assert (color >= 0).all() and (color <= 1).all()

    print("All tests passed!")
