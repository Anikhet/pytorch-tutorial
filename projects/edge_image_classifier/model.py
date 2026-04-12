"""
Model definitions for the edge image classifier pipeline.

Defines two CNN architectures for MNIST classification:
- TeacherCNN: A larger, more accurate model used as the knowledge source.
- StudentCNN: A smaller, faster model that learns from the teacher.

The student is designed to be small enough for edge deployment after
further compression (pruning + quantization).
"""

import torch
import torch.nn as nn


class TeacherCNN(nn.Module):
    """
    Larger CNN for MNIST classification (the 'teacher' in distillation).

    Architecture: 3 conv layers (32 -> 64 -> 128 filters) followed by
    two fully connected layers. Uses ReLU activations and max pooling.

    Input: (batch, 1, 28, 28) -- single-channel MNIST images.
    Output: (batch, 10) -- logits for 10 digit classes.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # After two 2x2 max pools: 28 -> 14 -> 7, so spatial dim is 7x7
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        """Forward pass through feature extractor and classifier."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class StudentCNN(nn.Module):
    """
    Smaller CNN for MNIST classification (the 'student' in distillation).

    Architecture: 2 conv layers (16 -> 32 filters) followed by two
    fully connected layers. Roughly 10x fewer parameters than TeacherCNN.

    Input: (batch, 1, 28, 28) -- single-channel MNIST images.
    Output: (batch, 10) -- logits for 10 digit classes.
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
        # After two 2x2 max pools: 28 -> 14 -> 7, so spatial dim is 7x7
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        """Forward pass through feature extractor and classifier."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_model_info(model):
    """
    Return parameter count and model size for a given PyTorch model.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        A dict with 'param_count' (int) and 'size_kb' (float).
        Size is estimated as 4 bytes per FP32 parameter.
    """
    param_count = sum(p.numel() for p in model.parameters())
    # Each FP32 param = 4 bytes
    size_kb = (param_count * 4) / 1024
    return {
        "param_count": param_count,
        "size_kb": round(size_kb, 2),
    }
