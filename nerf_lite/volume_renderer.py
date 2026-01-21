"""
Volume rendering for NeRF Lite.

This module implements the volume rendering equation that converts
density and color values along rays into final pixel colors.

The Volume Rendering Equation:
    C(r) = sum_{i=1}^{N} T_i * alpha_i * c_i

where:
    - T_i = prod_{j=1}^{i-1} (1 - alpha_j)  is the transmittance
    - alpha_i = 1 - exp(-sigma_i * delta_i) is the opacity
    - delta_i = t_{i+1} - t_i is the distance between samples
    - c_i is the color at sample i

Intuition:
    Each sample contributes color weighted by:
    1. How opaque it is (alpha_i) - dense regions contribute more
    2. How much light wasn't blocked earlier (T_i) - front samples dominate

Reference:
    Max, "Optical Models for Direct Volume Rendering", 1995
    Mildenhall et al., "NeRF", 2020
"""

from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def compute_alpha_from_density(
    density: torch.Tensor,
    deltas: torch.Tensor
) -> torch.Tensor:
    """
    Convert volume density to alpha (opacity) values.

    Uses the exponential transmittance model:
        alpha = 1 - exp(-sigma * delta)

    This represents the probability that a ray is absorbed or scattered
    within a segment of length delta with density sigma.

    Args:
        density: Volume density values [N_rays, N_samples]
        deltas: Distances between samples [N_rays, N_samples]

    Returns:
        alpha: Opacity values in [0, 1], shape [N_rays, N_samples]
    """
    return 1.0 - torch.exp(-density * deltas)


def compute_transmittance(alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute transmittance (visibility) from opacity values.

    Transmittance T_i is the probability that light reaches sample i
    without being absorbed by earlier samples:
        T_i = prod_{j=1}^{i-1} (1 - alpha_j)

    The first sample always has T_1 = 1 (nothing blocks it).

    Args:
        alpha: Opacity values [N_rays, N_samples]

    Returns:
        transmittance: Visibility weights [N_rays, N_samples]
    """
    # Compute cumulative product of (1 - alpha)
    # Shift by 1 to exclude current sample (T_i depends on j < i)
    one_minus_alpha = 1.0 - alpha + 1e-10  # Small epsilon for stability

    # Exclusive cumulative product: T_i = prod_{j<i}(1 - alpha_j)
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones_like(alpha[..., :1]),  # T_1 = 1
            one_minus_alpha[..., :-1]
        ], dim=-1),
        dim=-1
    )

    return transmittance


def volume_render(
    density: torch.Tensor,
    color: torch.Tensor,
    z_vals: torch.Tensor,
    ray_directions: torch.Tensor,
    white_background: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render pixel colors from density and color along rays.

    This is the core of NeRF: converting 3D scene representation
    to 2D images via numerical integration.

    Algorithm:
    1. Compute distances between adjacent samples (delta)
    2. Convert density to opacity (alpha)
    3. Compute transmittance (cumulative transparency)
    4. Weight colors by transmittance * alpha
    5. Sum weighted colors along each ray

    Args:
        density: Volume density at samples [N_rays, N_samples, 1]
        color: RGB color at samples [N_rays, N_samples, 3]
        z_vals: Depth values along rays [N_rays, N_samples]
        ray_directions: Ray direction vectors [N_rays, 3]
        white_background: If True, add white for transparent regions

    Returns:
        rgb: Rendered pixel colors [N_rays, 3]
        depth: Expected depth (weighted average) [N_rays]
        weights: Contribution weights [N_rays, N_samples]
    """
    # Compute distances between samples
    # delta_i = z_{i+1} - z_i
    deltas = z_vals[..., 1:] - z_vals[..., :-1]

    # Last delta: use a large value (infinity approximation)
    delta_last = torch.full_like(deltas[..., :1], 1e10)
    deltas = torch.cat([deltas, delta_last], dim=-1)

    # Scale by ray direction magnitude (for non-unit direction vectors)
    ray_norm = ray_directions.norm(dim=-1, keepdim=True)
    deltas = deltas * ray_norm

    # Convert density to alpha
    density_squeezed = density.squeeze(-1)  # [N_rays, N_samples]
    alpha = compute_alpha_from_density(density_squeezed, deltas)

    # Compute transmittance
    transmittance = compute_transmittance(alpha)

    # Compute weights: w_i = T_i * alpha_i
    weights = transmittance * alpha

    # Accumulate color: C = sum(w_i * c_i)
    rgb = torch.sum(weights.unsqueeze(-1) * color, dim=-2)

    # Compute expected depth: D = sum(w_i * z_i)
    depth = torch.sum(weights * z_vals, dim=-1)

    # Handle background
    if white_background:
        # Accumulated alpha (total opacity)
        acc = weights.sum(dim=-1, keepdim=True)
        # Add white color for transparent regions
        rgb = rgb + (1.0 - acc) * 1.0

    return rgb, depth, weights


def render_rays(
    model: torch.nn.Module,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    points: torch.Tensor,
    z_vals: torch.Tensor,
    chunk_size: int = 4096,
    white_background: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render rays using a NeRF model.

    Processes points in chunks to avoid GPU memory issues.

    Args:
        model: NeRF model with forward(positions, directions)
        ray_origins: Ray origins [N_rays, 3]
        ray_directions: Ray directions [N_rays, 3]
        points: Sample points [N_rays, N_samples, 3]
        z_vals: Depth values [N_rays, N_samples]
        chunk_size: Points to process at once
        white_background: Use white background

    Returns:
        rgb: Rendered colors [N_rays, 3]
        depth: Depth map [N_rays]
        weights: Sample weights [N_rays, N_samples]
    """
    N_rays, N_samples = points.shape[:2]

    # Flatten points for batch processing
    points_flat = points.reshape(-1, 3)

    # Expand directions to match points
    # Each ray's direction applies to all its samples
    directions_flat = ray_directions.unsqueeze(1).expand(-1, N_samples, -1)
    directions_flat = directions_flat.reshape(-1, 3)
    directions_flat = F.normalize(directions_flat, dim=-1)

    # Process in chunks to avoid OOM
    all_density = []
    all_color = []

    for i in range(0, points_flat.shape[0], chunk_size):
        chunk_points = points_flat[i:i + chunk_size]
        chunk_dirs = directions_flat[i:i + chunk_size]

        with torch.no_grad() if not model.training else torch.enable_grad():
            density, color = model(chunk_points, chunk_dirs)

        all_density.append(density)
        all_color.append(color)

    # Concatenate and reshape
    density = torch.cat(all_density, dim=0).reshape(N_rays, N_samples, 1)
    color = torch.cat(all_color, dim=0).reshape(N_rays, N_samples, 3)

    # Volume render
    return volume_render(density, color, z_vals, ray_directions, white_background)


def render_image(
    model: torch.nn.Module,
    intrinsics,
    pose,
    near: float,
    far: float,
    num_samples: int,
    chunk_size: int = 1024,
    white_background: bool = True,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render a complete image from a camera viewpoint.

    Convenience function that handles ray generation, point sampling,
    and volume rendering for a full image.

    Args:
        model: NeRF model
        intrinsics: CameraIntrinsics object
        pose: CameraPose object
        near: Near plane distance
        far: Far plane distance
        num_samples: Samples per ray
        chunk_size: Rays to process at once
        white_background: Use white background
        device: Torch device

    Returns:
        image: Rendered RGB image [H, W, 3]
        depth: Depth map [H, W]
    """
    from ray_utils import generate_rays, sample_points_along_rays

    H, W = intrinsics.height, intrinsics.width

    # Generate rays for all pixels
    ray_origins, ray_directions = generate_rays(intrinsics, pose, device)

    # Process in chunks to avoid OOM
    all_rgb = []
    all_depth = []

    for i in range(0, ray_origins.shape[0], chunk_size):
        chunk_origins = ray_origins[i:i + chunk_size]
        chunk_directions = ray_directions[i:i + chunk_size]

        # Sample points along rays
        points, z_vals = sample_points_along_rays(
            chunk_origins, chunk_directions,
            near, far, num_samples,
            stratified=False,  # Deterministic for inference
            device=device
        )

        # Render
        rgb, depth, _ = render_rays(
            model, chunk_origins, chunk_directions,
            points, z_vals,
            chunk_size=chunk_size * num_samples,
            white_background=white_background
        )

        all_rgb.append(rgb)
        all_depth.append(depth)

    # Reshape to image
    rgb = torch.cat(all_rgb, dim=0).reshape(H, W, 3)
    depth = torch.cat(all_depth, dim=0).reshape(H, W)

    return rgb, depth


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(max_val^2 / MSE)

    Higher is better. Typical values for good reconstruction: 25-35 dB.

    Args:
        pred: Predicted image
        target: Ground truth image
        max_val: Maximum pixel value

    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(pred, target)
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr


if __name__ == "__main__":
    # Test volume rendering
    print("Testing volume renderer...")

    device = "cpu"
    N_rays = 100
    N_samples = 64

    # Create synthetic test data
    # Uniform density with increasing values
    density = torch.ones(N_rays, N_samples, 1) * 0.1
    # Gradient color from red to blue along ray
    color = torch.zeros(N_rays, N_samples, 3)
    color[..., 0] = torch.linspace(1, 0, N_samples).unsqueeze(0)  # R
    color[..., 2] = torch.linspace(0, 1, N_samples).unsqueeze(0)  # B

    z_vals = torch.linspace(2.0, 6.0, N_samples).unsqueeze(0).expand(N_rays, -1)
    ray_directions = torch.tensor([[0, 0, -1.0]]).expand(N_rays, -1)

    # Render
    rgb, depth, weights = volume_render(
        density, color, z_vals, ray_directions, white_background=True
    )

    print(f"RGB shape: {rgb.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    print(f"Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"Weights shape: {weights.shape}, sum: {weights.sum(dim=-1).mean():.3f}")

    assert rgb.shape == (N_rays, 3)
    assert depth.shape == (N_rays,)
    assert weights.shape == (N_rays, N_samples)

    # Check weights sum to approximately 1 (some transmits through)
    weight_sum = weights.sum(dim=-1)
    print(f"Weight sum range: [{weight_sum.min():.3f}, {weight_sum.max():.3f}]")

    # Test with high density (should be opaque)
    density_high = torch.ones(N_rays, N_samples, 1) * 100.0
    rgb_opaque, _, weights_opaque = volume_render(
        density_high, color, z_vals, ray_directions, white_background=False
    )
    print(f"High density weight sum: {weights_opaque.sum(dim=-1).mean():.3f}")
    # First sample should dominate
    assert weights_opaque[:, 0].mean() > 0.9

    # Test PSNR
    pred = torch.rand(10, 10, 3)
    target = pred.clone()
    psnr_perfect = compute_psnr(pred, target)
    print(f"PSNR (identical): {psnr_perfect.item():.1f} dB")
    assert psnr_perfect > 100  # Should be very high for identical images

    target_noisy = pred + torch.randn_like(pred) * 0.1
    target_noisy = target_noisy.clamp(0, 1)
    psnr_noisy = compute_psnr(pred, target_noisy)
    print(f"PSNR (noisy): {psnr_noisy.item():.1f} dB")

    print("All volume renderer tests passed!")
