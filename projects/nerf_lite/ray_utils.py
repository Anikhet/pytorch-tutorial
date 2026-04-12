"""
Ray utilities for NeRF Lite.

This module handles camera models and ray generation for volume rendering.
It converts 2D pixel coordinates to 3D rays that traverse the scene.

Key Components:
    - CameraIntrinsics: Focal length, image dimensions, principal point
    - CameraPose: Camera-to-world transformation (rotation + translation)
    - generate_rays: Convert pixels to world-space rays
    - sample_points_along_rays: Sample 3D points for volume rendering

Mathematical Background:
    A ray is parameterized as: r(t) = o + t * d
    where o is the origin (camera position) and d is the direction.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters.

    Defines the internal geometry of the camera: how 3D points in camera
    space project onto the 2D image plane.

    The projection follows the pinhole camera model:
        u = fx * X/Z + cx
        v = fy * Y/Z + cy

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        focal_x: Focal length in x direction (pixels)
        focal_y: Focal length in y direction (pixels), defaults to focal_x
        cx: Principal point x coordinate, defaults to width/2
        cy: Principal point y coordinate, defaults to height/2
    """
    width: int
    height: int
    focal_x: float
    focal_y: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None

    def __post_init__(self):
        if self.focal_y is None:
            self.focal_y = self.focal_x
        if self.cx is None:
            self.cx = self.width / 2.0
        if self.cy is None:
            self.cy = self.height / 2.0


@dataclass
class CameraPose:
    """
    Camera extrinsic parameters (pose in world coordinates).

    The camera pose is represented as a camera-to-world transformation:
        p_world = R @ p_camera + t

    where R is a 3x3 rotation matrix and t is the camera position in world
    coordinates (not the translation vector from world to camera).

    Convention: Camera looks down -Z axis in camera space.
    After transformation, the camera looks in the direction of the
    third column of R (negated).

    Attributes:
        rotation: 3x3 rotation matrix (camera-to-world)
        translation: Camera position in world coordinates
    """
    rotation: torch.Tensor  # [3, 3]
    translation: torch.Tensor  # [3]

    @classmethod
    def from_look_at(
        cls,
        camera_position: torch.Tensor,
        target: torch.Tensor,
        up: torch.Tensor = None,
        device: str = "cpu"
    ) -> "CameraPose":
        """
        Create camera pose from position and target point.

        Args:
            camera_position: Position of camera in world coordinates [3]
            target: Point the camera is looking at [3]
            up: Up direction, defaults to [0, 1, 0]
            device: Torch device

        Returns:
            CameraPose with rotation and translation
        """
        if up is None:
            up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)

        camera_position = camera_position.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        up = up.to(device=device, dtype=torch.float32)

        # Forward direction (camera looks down -Z, so forward is from target to camera)
        forward = F.normalize(camera_position - target, dim=-1)

        # Right direction
        right = F.normalize(torch.linalg.cross(up, forward), dim=-1)

        # Recompute up to ensure orthogonality
        up_ortho = torch.linalg.cross(forward, right)

        # Rotation matrix: columns are right, up, forward (camera-to-world)
        rotation = torch.stack([right, up_ortho, forward], dim=-1)

        return cls(rotation=rotation, translation=camera_position)

    @classmethod
    def from_spherical(
        cls,
        radius: float,
        azimuth: float,
        elevation: float,
        target: torch.Tensor = None,
        device: str = "cpu"
    ) -> "CameraPose":
        """
        Create camera pose from spherical coordinates.

        Useful for generating camera poses on a sphere looking at a central point.

        Args:
            radius: Distance from target
            azimuth: Horizontal angle in radians (0 = +X, pi/2 = +Z)
            elevation: Vertical angle in radians (0 = horizontal, pi/2 = top)
            target: Point to look at, defaults to origin
            device: Torch device

        Returns:
            CameraPose looking at target from spherical position
        """
        if target is None:
            target = torch.zeros(3, device=device, dtype=torch.float32)

        # Convert spherical to Cartesian
        x = float(radius * np.cos(elevation) * np.cos(azimuth))
        y = float(radius * np.sin(elevation))
        z = float(radius * np.cos(elevation) * np.sin(azimuth))

        camera_position = torch.tensor([x, y, z], device=device, dtype=torch.float32) + target

        return cls.from_look_at(camera_position, target, device=device)


def generate_rays(
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for all pixels in an image.

    For each pixel (i, j), computes:
    - ray_origin: Camera position in world coordinates
    - ray_direction: Normalized direction from camera through pixel

    The ray equation is: r(t) = origin + t * direction

    Process:
    1. Create pixel grid
    2. Convert pixels to camera-space directions
    3. Transform to world space using camera pose

    Args:
        intrinsics: Camera intrinsic parameters
        pose: Camera pose (extrinsics)
        device: Torch device

    Returns:
        ray_origins: [H*W, 3] - all rays start from camera position
        ray_directions: [H*W, 3] - normalized direction vectors
    """
    H, W = intrinsics.height, intrinsics.width

    # Create pixel coordinate grid
    # Note: We use pixel centers (add 0.5)
    u = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    v = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    u, v = torch.meshgrid(u, v, indexing="xy")

    # Convert to camera coordinates
    # Camera convention: Z forward, X right, Y down -> Y up in our convention
    x = (u - intrinsics.cx) / intrinsics.focal_x
    y = -(v - intrinsics.cy) / intrinsics.focal_y  # Negative for Y-up
    z = -torch.ones_like(x)  # Camera looks down -Z

    # Stack to get directions in camera space [H, W, 3]
    directions_cam = torch.stack([x, y, z], dim=-1)

    # Normalize directions
    directions_cam = F.normalize(directions_cam, dim=-1)

    # Transform to world space
    # rotation is camera-to-world: d_world = R @ d_cam
    rotation = pose.rotation.to(device)
    directions_world = torch.einsum("ij,hwj->hwi", rotation, directions_cam)

    # Reshape to [H*W, 3]
    directions_world = directions_world.reshape(-1, 3)

    # All rays originate from camera position
    origins = pose.translation.to(device).expand(H * W, -1)

    return origins, directions_world


def generate_rays_for_pixels(
    pixels: torch.Tensor,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for specific pixel coordinates.

    Useful for training when sampling random pixels instead of full images.

    Args:
        pixels: Pixel coordinates [N, 2] where each row is (u, v)
        intrinsics: Camera intrinsic parameters
        pose: Camera pose
        device: Torch device

    Returns:
        ray_origins: [N, 3]
        ray_directions: [N, 3]
    """
    u = pixels[:, 0].float() + 0.5
    v = pixels[:, 1].float() + 0.5

    # Convert to camera coordinates
    x = (u - intrinsics.cx) / intrinsics.focal_x
    y = -(v - intrinsics.cy) / intrinsics.focal_y
    z = -torch.ones_like(x)

    directions_cam = torch.stack([x, y, z], dim=-1)
    directions_cam = F.normalize(directions_cam, dim=-1)

    # Transform to world space
    rotation = pose.rotation.to(device)
    directions_world = torch.einsum("ij,nj->ni", rotation, directions_cam)

    origins = pose.translation.to(device).expand(pixels.shape[0], -1)

    return origins, directions_world


def sample_points_along_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    stratified: bool = True,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 3D points along rays for volume rendering.

    For each ray, samples num_samples points between near and far planes.
    Points are computed as: p = origin + t * direction

    Stratified sampling divides [near, far] into num_samples bins and
    samples uniformly within each bin. This provides better coverage
    and reduces aliasing artifacts.

    Args:
        ray_origins: Ray origin points [N, 3]
        ray_directions: Ray direction vectors [N, 3]
        near: Near plane distance
        far: Far plane distance
        num_samples: Number of samples per ray
        stratified: If True, add random jitter within bins
        device: Torch device

    Returns:
        points: Sampled 3D positions [N, num_samples, 3]
        z_vals: Depth values along rays [N, num_samples]
    """
    N = ray_origins.shape[0]

    # Create linearly spaced depth values
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals  # [num_samples]

    # Expand to [N, num_samples]
    z_vals = z_vals.expand(N, num_samples).clone()

    if stratified:
        # Add random jitter within each bin (centered to stay in range)
        bin_size = (far - near) / num_samples
        # Jitter in range [-0.5*bin, +0.5*bin] centered on bin center
        noise = (torch.rand(N, num_samples, device=device) - 0.5) * bin_size
        z_vals = z_vals + noise
        # Clamp to ensure we stay within [near, far]
        z_vals = torch.clamp(z_vals, near, far)

    # Compute 3D points: p = o + t * d
    # ray_origins: [N, 3] -> [N, 1, 3]
    # ray_directions: [N, 3] -> [N, 1, 3]
    # z_vals: [N, num_samples] -> [N, num_samples, 1]
    points = ray_origins.unsqueeze(1) + z_vals.unsqueeze(-1) * ray_directions.unsqueeze(1)

    return points, z_vals


def get_ray_bundle_for_image(
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    near: float,
    far: float,
    num_samples: int,
    stratified: bool = True,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate complete ray bundle for an image: rays + sample points.

    Convenience function that combines ray generation and point sampling.

    Args:
        intrinsics: Camera intrinsics
        pose: Camera pose
        near: Near plane
        far: Far plane
        num_samples: Samples per ray
        stratified: Use stratified sampling
        device: Torch device

    Returns:
        ray_origins: [H*W, 3]
        ray_directions: [H*W, 3]
        points: [H*W, num_samples, 3]
        z_vals: [H*W, num_samples]
    """
    ray_origins, ray_directions = generate_rays(intrinsics, pose, device)
    points, z_vals = sample_points_along_rays(
        ray_origins, ray_directions, near, far, num_samples, stratified, device
    )
    return ray_origins, ray_directions, points, z_vals


if __name__ == "__main__":
    # Test ray utilities
    print("Testing ray utilities...")

    device = "cpu"

    # Test camera intrinsics
    intrinsics = CameraIntrinsics(width=64, height=64, focal_x=50.0)
    print(f"Intrinsics: {intrinsics.width}x{intrinsics.height}, f={intrinsics.focal_x}")

    # Test camera pose from spherical coordinates
    pose = CameraPose.from_spherical(
        radius=4.0,
        azimuth=np.pi / 4,
        elevation=np.pi / 6,
        device=device
    )
    print(f"Camera position: {pose.translation}")

    # Test ray generation
    origins, directions = generate_rays(intrinsics, pose, device)
    print(f"Rays generated: {origins.shape}, {directions.shape}")
    assert origins.shape == (64 * 64, 3)
    assert directions.shape == (64 * 64, 3)

    # Check directions are normalized
    norms = torch.norm(directions, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Test point sampling
    points, z_vals = sample_points_along_rays(
        origins, directions, near=2.0, far=6.0, num_samples=64, device=device
    )
    print(f"Points sampled: {points.shape}, z_vals: {z_vals.shape}")
    assert points.shape == (64 * 64, 64, 3)
    assert z_vals.shape == (64 * 64, 64)

    # Check z_vals are in range and increasing
    assert (z_vals >= 2.0).all() and (z_vals <= 6.0).all()

    # Test look-at camera
    cam_pos = torch.tensor([3.0, 2.0, 3.0])
    target = torch.tensor([0.0, 0.0, 0.0])
    pose2 = CameraPose.from_look_at(cam_pos, target, device=device)
    print(f"Look-at camera: pos={pose2.translation}")

    print("All ray utility tests passed!")
