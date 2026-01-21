"""
Diffusion Utilities - Noise scheduling and DDIM sampling.

This module implements:
1. Cosine noise schedule (better signal preservation than linear)
2. NoiseScheduler for adding/removing noise
3. DDIM sampler for fast inference (20 steps vs 1000)

Mathematical foundations from:
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
"""

import torch
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from 'Improved Denoising Diffusion Probabilistic Models'.

    Preserves more signal in early timesteps compared to linear schedule,
    leading to better image quality.

    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent beta from being too small at t=0

    Returns:
        Beta values for each timestep [timesteps]
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    # Compute cumulative alphas using cosine
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # Compute betas from alphas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # Clip to prevent numerical issues
    return torch.clip(betas, 0.0001, 0.9999)


class NoiseScheduler:
    """
    Noise scheduler for diffusion models.

    Handles the forward process (adding noise) and provides values
    needed for the reverse process (removing noise).
    """

    def __init__(self, num_timesteps: int = 200, device: str = "cpu"):
        """
        Initialize the noise scheduler.

        Args:
            num_timesteps: Total number of diffusion steps
            device: Device to store tensors on
        """
        self.num_timesteps = num_timesteps
        self.device = device

        # Compute schedule values
        self.betas = cosine_beta_schedule(num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Pre-compute useful values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # For posterior computation (used in DDPM, not DDIM)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to clean images (forward diffusion process).

        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x_0: Clean images [batch, C, H, W]
            noise: Gaussian noise [batch, C, H, W]
            t: Timesteps [batch]

        Returns:
            Noisy images x_t [batch, C, H, W]
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def get_alphas_for_timestep(self, t: int) -> tuple[float, float]:
        """Get alpha values for a specific timestep."""
        return self.alphas_cumprod[t].item(), self.sqrt_one_minus_alphas_cumprod[t].item()


@torch.inference_mode()
def ddim_sample(
    model,
    sketch: torch.Tensor,
    scheduler: NoiseScheduler,
    num_inference_steps: int = 20,
    eta: float = 0.0,
    show_progress: bool = True
) -> torch.Tensor:
    """
    DDIM sampling for fast inference.

    DDIM is deterministic (when eta=0) and allows fewer steps than DDPM
    while maintaining quality. 20 steps achieves similar quality to
    1000 steps of DDPM.

    Args:
        model: TinySketchUNet model
        sketch: Conditioning sketch [batch, 3, H, W]
        scheduler: NoiseScheduler instance
        num_inference_steps: Number of denoising steps (20 recommended)
        eta: Noise scale (0 = deterministic, 1 = DDPM-like stochasticity)
        show_progress: Whether to show tqdm progress bar

    Returns:
        Generated images [batch, 3, H, W]
    """
    device = sketch.device
    batch_size = sketch.shape[0]
    img_size = sketch.shape[-1]

    # Start from pure noise
    x_t = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # Create timestep sequence (evenly spaced)
    step_ratio = scheduler.num_timesteps // num_inference_steps
    timesteps = list(range(0, scheduler.num_timesteps, step_ratio))[::-1]

    iterator = tqdm(timesteps, desc="Generating") if show_progress else timesteps

    for i, t in enumerate(iterator):
        # Create timestep tensor
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Concatenate noisy image with sketch condition
        model_input = torch.cat([x_t, sketch], dim=1)

        # Predict noise
        noise_pred = model(model_input, t_tensor)

        # Get alpha values
        alpha_t = scheduler.alphas_cumprod[t]
        alpha_t_prev = scheduler.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)

        # DDIM update
        # Predict x_0 from x_t and predicted noise
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        # Clip for stability
        pred_x0 = pred_x0.clamp(-1, 1)

        # Direction pointing to x_t
        direction = torch.sqrt(1 - alpha_t_prev) * noise_pred

        # Optional noise (eta controls stochasticity)
        if eta > 0 and i < len(timesteps) - 1:
            sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            noise = torch.randn_like(x_t)
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + direction + sigma * noise
        else:
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + direction

    return x_t


def preprocess_sketch(
    sketch_array,
    size: int = 32,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Preprocess a user-drawn sketch for model input.

    Handles various input formats from Gradio Sketchpad and converts
    to normalized tensor suitable for the diffusion model.

    Args:
        sketch_array: Input from Gradio (numpy array or dict)
        size: Target size (32 for CPU-optimized model)
        device: Target device

    Returns:
        Preprocessed sketch tensor [3, size, size] normalized to [-1, 1]
    """
    import numpy as np
    from PIL import Image

    # Handle different Gradio output formats
    if isinstance(sketch_array, dict):
        # Newer Gradio versions return a dict with 'composite' key
        if 'composite' in sketch_array:
            sketch_array = sketch_array['composite']
        elif 'image' in sketch_array:
            sketch_array = sketch_array['image']

    # Convert to PIL Image
    if isinstance(sketch_array, np.ndarray):
        # Handle RGBA (4 channels) from Sketchpad
        if sketch_array.ndim == 3 and sketch_array.shape[2] == 4:
            # Use alpha channel to create black lines on white background
            alpha = sketch_array[:, :, 3] / 255.0
            # Create grayscale from RGB, inverted (black lines -> white, white bg -> black)
            rgb = sketch_array[:, :, :3].mean(axis=2) / 255.0
            # Combine: where alpha is high, use the drawn content
            sketch_gray = 1.0 - alpha * (1.0 - rgb)
            sketch_array = (sketch_gray * 255).astype(np.uint8)
        elif sketch_array.ndim == 3:
            # RGB image
            sketch_array = np.mean(sketch_array, axis=2).astype(np.uint8)
        img = Image.fromarray(sketch_array, mode='L')
    else:
        img = sketch_array

    # Resize
    img = img.resize((size, size), Image.Resampling.BILINEAR)

    # Convert to tensor and normalize to [-1, 1]
    tensor = torch.from_numpy(np.array(img)).float() / 255.0
    tensor = tensor * 2 - 1  # [0, 1] -> [-1, 1]

    # Expand to 3 channels
    tensor = tensor.unsqueeze(0).expand(3, -1, -1)

    return tensor.to(device)


# Test when run directly
if __name__ == "__main__":
    print("Testing diffusion utilities...")

    # Test noise scheduler
    scheduler = NoiseScheduler(num_timesteps=200)
    print(f"Beta range: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
    print(f"Alpha_cumprod range: [{scheduler.alphas_cumprod.min():.4f}, {scheduler.alphas_cumprod.max():.4f}]")

    # Test adding noise
    x_0 = torch.randn(2, 3, 32, 32)
    noise = torch.randn_like(x_0)
    t = torch.tensor([0, 199])

    x_t = scheduler.add_noise(x_0, noise, t)
    print(f"Noisy image shape: {x_t.shape}")

    print("All tests passed!")
