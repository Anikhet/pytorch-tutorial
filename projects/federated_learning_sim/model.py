"""
Shared CNN model for federated learning.

All clients and the server use the same model architecture (SimpleCNN).
Only weights are communicated -- never the architecture or data.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A small CNN for MNIST digit classification.

    Architecture:
        Conv2d(1, 16, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Linear(32 * 7 * 7, 128) -> ReLU
        Linear(128, 10)

    Input:  (batch, 1, 28, 28) grayscale images
    Output: (batch, 10) class logits
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """Forward pass: features -> flatten -> classifier."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model_info():
    """
    Print a summary of the SimpleCNN model.

    Returns:
        dict with 'total_params' and 'layer_names' keys.
    """
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    layer_names = [name for name, _ in model.named_parameters()]

    return {
        "total_params": total_params,
        "layer_names": layer_names,
    }


if __name__ == "__main__":
    info = get_model_info()
    print(f"SimpleCNN - Total parameters: {info['total_params']:,}")
    for name in info["layer_names"]:
        print(f"  {name}")
