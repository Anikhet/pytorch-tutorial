"""
Simple neural networks and datasets for hyperparameter visualization.

Provides fast-training models to demonstrate effects of different
hyperparameters like learning rate, batch size, and optimizer choice.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for classification.

    Small enough to train quickly, large enough to show
    interesting hyperparameter effects.
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_sizes: Tuple[int, ...] = (32, 32),
        output_size: int = 2,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        super().__init__()

        # Select activation
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU
        }
        act_fn = activations.get(activation, nn.ReLU)

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepMLP(nn.Module):
    """
    Deeper MLP to show effects of depth and batch normalization.
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 5,
        output_size: int = 2,
        use_batch_norm: bool = False,
        activation: str = "relu"
    ):
        super().__init__()

        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU
        }
        act_fn = activations.get(activation, nn.ReLU)

        layers = [nn.Linear(input_size, hidden_size), act_fn()]
        if use_batch_norm:
            layers.insert(1, nn.BatchNorm1d(hidden_size))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_fn())

        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def generate_spiral_data(
    n_samples: int = 1000,
    n_classes: int = 2,
    noise: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate spiral classification dataset.

    Classic dataset that requires non-linear decision boundaries.
    Good for visualizing training dynamics.
    """
    X = []
    y = []

    samples_per_class = n_samples // n_classes

    for class_idx in range(n_classes):
        r = np.linspace(0.2, 1, samples_per_class)
        theta = np.linspace(
            class_idx * 4,
            (class_idx + 1) * 4,
            samples_per_class
        ) + np.random.randn(samples_per_class) * noise

        x1 = r * np.sin(theta)
        x2 = r * np.cos(theta)

        X.append(np.column_stack([x1, x2]))
        y.append(np.full(samples_per_class, class_idx))

    X = np.vstack(X)
    y = np.concatenate(y)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return torch.FloatTensor(X), torch.LongTensor(y)


def generate_circles_data(
    n_samples: int = 1000,
    noise: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate concentric circles dataset.

    Tests ability to learn circular decision boundaries.
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Outer circle
    theta_out = np.random.uniform(0, 2 * np.pi, n_samples_out)
    r_out = 1.0 + np.random.randn(n_samples_out) * noise
    X_out = np.column_stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])

    # Inner circle
    theta_in = np.random.uniform(0, 2 * np.pi, n_samples_in)
    r_in = 0.5 + np.random.randn(n_samples_in) * noise
    X_in = np.column_stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])

    X = np.vstack([X_out, X_in])
    y = np.array([0] * n_samples_out + [1] * n_samples_in)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return torch.FloatTensor(X), torch.LongTensor(y)


def generate_moons_data(
    n_samples: int = 1000,
    noise: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two interleaving half circles (moons) dataset.
    """
    n_samples_per_moon = n_samples // 2

    # First moon
    theta1 = np.linspace(0, np.pi, n_samples_per_moon)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)

    # Second moon
    theta2 = np.linspace(0, np.pi, n_samples - n_samples_per_moon)
    x2 = 1 - np.cos(theta2)
    y2 = 0.5 - np.sin(theta2)

    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])
    X += np.random.randn(*X.shape) * noise

    y = np.array([0] * n_samples_per_moon + [1] * (n_samples - n_samples_per_moon))

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return torch.FloatTensor(X), torch.LongTensor(y)


def generate_xor_data(
    n_samples: int = 1000,
    noise: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate XOR dataset - classic non-linearly separable problem.
    """
    X = np.random.randn(n_samples, 2) * 0.5

    # Assign XOR labels
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    # Shift to quadrants
    X[:, 0] += np.where(X[:, 0] > 0, 0.5, -0.5)
    X[:, 1] += np.where(X[:, 1] > 0, 0.5, -0.5)

    # Add noise
    X += np.random.randn(*X.shape) * noise

    return torch.FloatTensor(X), torch.LongTensor(y)


DATASETS = {
    "spiral": generate_spiral_data,
    "circles": generate_circles_data,
    "moons": generate_moons_data,
    "xor": generate_xor_data
}


# Test when run directly
if __name__ == "__main__":
    print("Testing neural networks and datasets...")

    # Test SimpleMLP
    model = SimpleMLP(input_size=2, hidden_sizes=(32, 32), output_size=2)
    print(f"SimpleMLP parameters: {model.count_parameters()}")

    x = torch.randn(16, 2)
    out = model(x)
    print(f"SimpleMLP output shape: {out.shape}")

    # Test datasets
    for name, gen_fn in DATASETS.items():
        X, y = gen_fn(n_samples=100)
        print(f"{name}: X shape {X.shape}, y shape {y.shape}, classes {y.unique().tolist()}")

    print("All tests passed!")
