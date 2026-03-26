"""
Differential privacy utilities for federated learning.

Implements the core building blocks of DP-SGD:
  1. Gradient clipping  - bounds each sample's influence
  2. Noise addition     - adds calibrated Gaussian noise
  3. Privacy accounting - tracks cumulative privacy loss (epsilon)

Key idea: By clipping gradients and adding noise, we ensure that
the model update does not reveal too much about any single training
example. The tradeoff is that more noise means better privacy but
slower convergence and lower final accuracy.
"""

import math

import torch


def clip_gradients(model, max_norm):
    """
    Clip all parameter gradients to have at most `max_norm` L2 norm.

    This bounds the maximum influence any single sample can have
    on the model update. Essential for differential privacy --
    without clipping, a single outlier could dominate the gradient.

    Args:
        model:    PyTorch model whose .grad attributes will be clipped.
        max_norm: Maximum allowed L2 norm for the gradient vector.

    Returns:
        The actual (pre-clip) gradient norm as a float.
    """
    # Compute the total L2 norm across all parameters
    total_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm_sq += param.grad.data.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm_sq)

    # Scale gradients down if they exceed the max norm
    clip_factor = max_norm / max(total_norm, max_norm)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.mul_(clip_factor)

    return total_norm


def add_noise(model, noise_multiplier, max_norm):
    """
    Add calibrated Gaussian noise to all parameter gradients.

    The noise standard deviation is: noise_multiplier * max_norm.
    Combined with gradient clipping, this provides differential
    privacy guarantees -- the noise masks individual contributions.

    Args:
        model:            PyTorch model whose .grad attributes get noise.
        noise_multiplier: Controls privacy level (higher = more private).
        max_norm:         The clipping bound (used to calibrate noise).
    """
    noise_std = noise_multiplier * max_norm
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad.data) * noise_std
            param.grad.data.add_(noise)


def compute_privacy_spent(noise_multiplier, num_rounds, num_samples, delta=1e-5):
    """
    Estimate the total privacy budget (epsilon) spent over training.

    Uses basic composition theorem (not advanced RDP accounting):
        epsilon_per_round ~ sqrt(2 * ln(1.25 / delta)) / noise_multiplier
        epsilon_total ~ epsilon_per_round * sqrt(num_rounds)

    The sqrt(num_rounds) factor comes from advanced composition,
    which gives tighter bounds than naive linear composition.

    NOTE: This is a simplified estimate. Production systems should
    use Renyi Differential Privacy (RDP) accountants for tighter bounds.

    Args:
        noise_multiplier: The noise scale used during training.
        num_rounds:       Total number of training rounds.
        num_samples:      Number of training samples (for reference).
        delta:            Probability of privacy failure (default 1e-5).

    Returns:
        Estimated epsilon (lower = more private).
    """
    if noise_multiplier <= 0:
        return float("inf")

    # Per-round epsilon via Gaussian mechanism
    epsilon_per_round = math.sqrt(2.0 * math.log(1.25 / delta)) / noise_multiplier

    # Advanced composition over multiple rounds
    epsilon_total = epsilon_per_round * math.sqrt(num_rounds)

    return epsilon_total
