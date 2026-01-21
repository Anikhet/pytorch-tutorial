"""
Neural networks with gradient tracking hooks.

Provides models instrumented for detailed monitoring:
- Gradient statistics per layer
- Activation statistics
- Weight distributions
- Architecture information
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class LayerStats:
    """Statistics for a single layer."""
    name: str
    param_count: int
    weight_mean: float = 0.0
    weight_std: float = 0.0
    weight_min: float = 0.0
    weight_max: float = 0.0
    grad_mean: float = 0.0
    grad_std: float = 0.0
    grad_min: float = 0.0
    grad_max: float = 0.0
    grad_norm: float = 0.0
    activation_mean: float = 0.0
    activation_std: float = 0.0


@dataclass
class ModelStats:
    """Container for all model statistics."""
    layer_stats: Dict[str, LayerStats] = field(default_factory=dict)
    total_params: int = 0
    total_grad_norm: float = 0.0
    weight_histograms: Dict[str, np.ndarray] = field(default_factory=dict)
    grad_histograms: Dict[str, np.ndarray] = field(default_factory=dict)
    activation_histograms: Dict[str, np.ndarray] = field(default_factory=dict)


class MonitoredMLP(nn.Module):
    """
    Multi-Layer Perceptron with built-in monitoring.

    Tracks gradients, activations, and weights for visualization.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: Tuple[int, ...] = (256, 128, 64),
        output_size: int = 10,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Activation functions
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU
        }
        act_fn = activations.get(activation, nn.ReLU)

        # Build layers with names for tracking
        self.layers = nn.ModuleDict()
        self.activations_dict = nn.ModuleDict()

        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers[f"fc{i+1}"] = nn.Linear(prev_size, hidden_size)
            self.activations_dict[f"act{i+1}"] = act_fn()
            if dropout > 0:
                self.activations_dict[f"drop{i+1}"] = nn.Dropout(dropout)
            prev_size = hidden_size

        self.layers["output"] = nn.Linear(prev_size, output_size)

        # Storage for monitoring
        self._activation_storage: Dict[str, torch.Tensor] = {}
        self._gradient_storage: Dict[str, torch.Tensor] = {}
        self._hooks: List = []

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def register_hooks(self):
        """Register forward and backward hooks for monitoring."""
        self.remove_hooks()

        for name, layer in self.layers.items():
            # Forward hook to capture activations
            hook = layer.register_forward_hook(
                lambda m, inp, out, n=name: self._save_activation(n, out)
            )
            self._hooks.append(hook)

            # Backward hook to capture gradients
            if layer.weight.requires_grad:
                hook = layer.weight.register_hook(
                    lambda grad, n=name: self._save_gradient(n, grad)
                )
                self._hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _save_activation(self, name: str, activation: torch.Tensor):
        """Store activation for monitoring."""
        self._activation_storage[name] = activation.detach()

    def _save_gradient(self, name: str, gradient: torch.Tensor):
        """Store gradient for monitoring."""
        self._gradient_storage[name] = gradient.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Pass through hidden layers
        for i in range(len(self.hidden_sizes)):
            x = self.layers[f"fc{i+1}"](x)
            x = self.activations_dict[f"act{i+1}"](x)
            if f"drop{i+1}" in self.activations_dict:
                x = self.activations_dict[f"drop{i+1}"](x)

        # Output layer
        x = self.layers["output"](x)
        return x

    def get_architecture_info(self) -> List[Dict]:
        """Get information about each layer for visualization."""
        info = []
        prev_size = self.input_size

        for i, hidden_size in enumerate(self.hidden_sizes):
            info.append({
                "name": f"fc{i+1}",
                "type": "Linear",
                "input_size": prev_size,
                "output_size": hidden_size,
                "params": prev_size * hidden_size + hidden_size
            })
            info.append({
                "name": f"act{i+1}",
                "type": "Activation",
                "input_size": hidden_size,
                "output_size": hidden_size,
                "params": 0
            })
            prev_size = hidden_size

        info.append({
            "name": "output",
            "type": "Linear",
            "input_size": prev_size,
            "output_size": self.output_size,
            "params": prev_size * self.output_size + self.output_size
        })

        return info

    def get_model_stats(self) -> ModelStats:
        """Collect comprehensive model statistics."""
        stats = ModelStats()
        stats.total_params = sum(p.numel() for p in self.parameters())

        # Calculate total gradient norm
        total_norm = 0.0

        for name, layer in self.layers.items():
            if not hasattr(layer, 'weight'):
                continue

            weight = layer.weight.data.cpu().numpy()
            layer_stats = LayerStats(
                name=name,
                param_count=layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0),
                weight_mean=float(np.mean(weight)),
                weight_std=float(np.std(weight)),
                weight_min=float(np.min(weight)),
                weight_max=float(np.max(weight))
            )

            # Weight histogram
            stats.weight_histograms[name] = weight.flatten()

            # Gradient statistics if available
            if name in self._gradient_storage:
                grad = self._gradient_storage[name].cpu().numpy()
                layer_stats.grad_mean = float(np.mean(grad))
                layer_stats.grad_std = float(np.std(grad))
                layer_stats.grad_min = float(np.min(grad))
                layer_stats.grad_max = float(np.max(grad))
                layer_stats.grad_norm = float(np.linalg.norm(grad))
                stats.grad_histograms[name] = grad.flatten()
                total_norm += layer_stats.grad_norm ** 2

            # Activation statistics if available
            if name in self._activation_storage:
                act = self._activation_storage[name].cpu().numpy()
                layer_stats.activation_mean = float(np.mean(act))
                layer_stats.activation_std = float(np.std(act))
                stats.activation_histograms[name] = act.flatten()

            stats.layer_stats[name] = layer_stats

        stats.total_grad_norm = float(np.sqrt(total_norm))
        return stats

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MonitoredCNN(nn.Module):
    """
    Convolutional Neural Network with monitoring for image classification.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        base_channels: int = 32
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # Convolutional layers
        self.conv_layers = nn.ModuleDict({
            "conv1": nn.Conv2d(input_channels, base_channels, 3, padding=1),
            "conv2": nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            "conv3": nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)
        })

        self.pool = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()

        # Calculate flattened size (assuming 28x28 input -> 3x3 after 3 pools)
        self.flat_size = base_channels * 4 * 3 * 3

        # Fully connected layers
        self.fc_layers = nn.ModuleDict({
            "fc1": nn.Linear(self.flat_size, 128),
            "fc2": nn.Linear(128, num_classes)
        })

        self._activation_storage: Dict[str, torch.Tensor] = {}
        self._gradient_storage: Dict[str, torch.Tensor] = {}
        self._hooks: List = []

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def register_hooks(self):
        """Register hooks for monitoring."""
        self.remove_hooks()

        all_layers = {**self.conv_layers, **self.fc_layers}
        for name, layer in all_layers.items():
            hook = layer.register_forward_hook(
                lambda m, inp, out, n=name: self._save_activation(n, out)
            )
            self._hooks.append(hook)

            if hasattr(layer, 'weight') and layer.weight.requires_grad:
                hook = layer.weight.register_hook(
                    lambda grad, n=name: self._save_gradient(n, grad)
                )
                self._hooks.append(hook)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _save_activation(self, name: str, activation: torch.Tensor):
        self._activation_storage[name] = activation.detach()

    def _save_gradient(self, name: str, gradient: torch.Tensor):
        self._gradient_storage[name] = gradient.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv layers
        x = self.pool(self.activation(self.conv_layers["conv1"](x)))
        x = self.pool(self.activation(self.conv_layers["conv2"](x)))
        x = self.pool(self.activation(self.conv_layers["conv3"](x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.activation(self.fc_layers["fc1"](x))
        x = self.fc_layers["fc2"](x)

        return x

    def get_architecture_info(self) -> List[Dict]:
        info = []
        for name, layer in self.conv_layers.items():
            info.append({
                "name": name,
                "type": "Conv2d",
                "input_size": layer.in_channels,
                "output_size": layer.out_channels,
                "params": layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            })
        for name, layer in self.fc_layers.items():
            info.append({
                "name": name,
                "type": "Linear",
                "input_size": layer.in_features,
                "output_size": layer.out_features,
                "params": layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            })
        return info

    def get_model_stats(self) -> ModelStats:
        """Collect model statistics."""
        stats = ModelStats()
        stats.total_params = sum(p.numel() for p in self.parameters())
        total_norm = 0.0

        all_layers = {**self.conv_layers, **self.fc_layers}
        for name, layer in all_layers.items():
            if not hasattr(layer, 'weight'):
                continue

            weight = layer.weight.data.cpu().numpy().flatten()
            layer_stats = LayerStats(
                name=name,
                param_count=layer.weight.numel(),
                weight_mean=float(np.mean(weight)),
                weight_std=float(np.std(weight)),
                weight_min=float(np.min(weight)),
                weight_max=float(np.max(weight))
            )

            stats.weight_histograms[name] = weight

            if name in self._gradient_storage:
                grad = self._gradient_storage[name].cpu().numpy().flatten()
                layer_stats.grad_mean = float(np.mean(grad))
                layer_stats.grad_std = float(np.std(grad))
                layer_stats.grad_min = float(np.min(grad))
                layer_stats.grad_max = float(np.max(grad))
                layer_stats.grad_norm = float(np.linalg.norm(grad))
                stats.grad_histograms[name] = grad
                total_norm += layer_stats.grad_norm ** 2

            if name in self._activation_storage:
                act = self._activation_storage[name].cpu().numpy().flatten()
                layer_stats.activation_mean = float(np.mean(act))
                layer_stats.activation_std = float(np.std(act))
                stats.activation_histograms[name] = act[:1000]  # Limit size

            stats.layer_stats[name] = layer_stats

        stats.total_grad_norm = float(np.sqrt(total_norm))
        return stats

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test when run directly
if __name__ == "__main__":
    print("Testing monitored networks...")

    # Test MLP
    mlp = MonitoredMLP(input_size=784, hidden_sizes=(256, 128), output_size=10)
    mlp.register_hooks()
    print(f"MLP parameters: {mlp.count_parameters():,}")

    x = torch.randn(32, 784)
    y = mlp(x)
    print(f"MLP output shape: {y.shape}")

    # Simulate backward pass
    loss = y.sum()
    loss.backward()

    stats = mlp.get_model_stats()
    print(f"Total gradient norm: {stats.total_grad_norm:.4f}")
    print(f"Layers tracked: {list(stats.layer_stats.keys())}")

    # Test CNN
    cnn = MonitoredCNN(input_channels=1, num_classes=10)
    cnn.register_hooks()
    print(f"\nCNN parameters: {cnn.count_parameters():,}")

    x = torch.randn(32, 1, 28, 28)
    y = cnn(x)
    print(f"CNN output shape: {y.shape}")

    print("\nArchitecture info:")
    for layer in mlp.get_architecture_info():
        print(f"  {layer['name']}: {layer['type']} ({layer['params']:,} params)")

    print("\nAll tests passed!")
