"""
Dataset utilities for NeRF Lite.

This module provides dataset classes for training NeRF:
1. SyntheticSceneDataset: Procedurally generated scenes (no download needed)
2. BlenderDataset: Load NeRF synthetic datasets (e.g., Blender scenes)

The synthetic dataset enables immediate experimentation without external data.
It generates simple geometric shapes with analytically computed ground truth.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from ray_utils import CameraIntrinsics, CameraPose, generate_rays


@dataclass
class Sphere:
    """A sphere in the scene."""
    center: torch.Tensor  # [3]
    radius: float
    color: torch.Tensor  # [3] RGB in [0, 1]


@dataclass
class SyntheticScene:
    """Configuration for synthetic scene generation."""
    num_spheres: int = 3
    sphere_radius_range: Tuple[float, float] = (0.2, 0.5)
    position_range: float = 1.0
    seed: int = 42


class SyntheticSceneDataset(Dataset):
    """
    Dataset with procedurally generated synthetic scenes.

    Creates a scene with colored spheres and renders ground truth images
    from multiple camera viewpoints. No external data download required.

    The scene consists of:
    - Multiple colored spheres of varying sizes
    - Camera poses sampled on a sphere looking at the origin
    - Analytically ray-traced ground truth images

    This enables rapid prototyping and debugging of NeRF implementations.

    Args:
        scene_config: Configuration for scene generation
        num_views: Number of training viewpoints
        image_size: Resolution of rendered images
        camera_distance: Distance of cameras from origin
        near: Near plane for ray sampling
        far: Far plane for ray sampling
        split: "train", "val", or "test"
        device: Torch device
    """

    def __init__(
        self,
        scene_config: Optional[SyntheticScene] = None,
        num_views: int = 100,
        image_size: int = 64,
        camera_distance: float = 4.0,
        near: float = 2.0,
        far: float = 6.0,
        split: str = "train",
        device: str = "cpu"
    ):
        self.config = scene_config or SyntheticScene()
        self.num_views = num_views
        self.image_size = image_size
        self.camera_distance = camera_distance
        self.near = near
        self.far = far
        self.split = split
        self.device = device

        # Set seed based on split for reproducibility
        split_seeds = {"train": 0, "val": 1000, "test": 2000}
        self.seed = self.config.seed + split_seeds.get(split, 0)

        # Generate scene geometry
        self.spheres = self._generate_scene()

        # Generate camera poses
        self.poses = self._generate_camera_poses()

        # Camera intrinsics
        self.intrinsics = CameraIntrinsics(
            width=image_size,
            height=image_size,
            focal_x=image_size * 0.8  # ~55 degree FOV
        )

        # Pre-render all ground truth images
        self.images = self._render_all_views()

    def _generate_scene(self) -> List[Sphere]:
        """Generate random spheres for the scene."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        spheres = []
        for i in range(self.config.num_spheres):
            # Random position
            center = torch.rand(3) * 2 * self.config.position_range - self.config.position_range

            # Random radius
            r_min, r_max = self.config.sphere_radius_range
            radius = np.random.uniform(r_min, r_max)

            # Random color (vibrant colors)
            color = torch.rand(3) * 0.7 + 0.3  # Avoid very dark colors

            spheres.append(Sphere(center=center, radius=radius, color=color))

        return spheres

    def _generate_camera_poses(self) -> List[CameraPose]:
        """Generate camera poses on a sphere around the scene."""
        np.random.seed(self.seed)

        poses = []
        for i in range(self.num_views):
            # Fibonacci sphere sampling for uniform distribution
            golden_ratio = (1 + np.sqrt(5)) / 2
            theta = 2 * np.pi * i / golden_ratio  # azimuth
            phi = np.arccos(1 - 2 * (i + 0.5) / self.num_views)  # polar

            # Convert to elevation (from horizontal)
            elevation = np.pi / 2 - phi

            # Clamp elevation to avoid gimbal lock
            elevation = np.clip(elevation, -np.pi / 3, np.pi / 3)

            pose = CameraPose.from_spherical(
                radius=self.camera_distance,
                azimuth=theta,
                elevation=elevation,
                device=self.device
            )
            poses.append(pose)

        return poses

    def _ray_sphere_intersection(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        sphere: Sphere
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ray-sphere intersection.

        Solves: |o + t*d - c|^2 = r^2
        Expands to: t^2|d|^2 + 2t*d.(o-c) + |o-c|^2 - r^2 = 0

        Returns:
            hit_mask: Boolean mask of rays that hit [N_rays]
            t_hit: Distance to intersection (inf if no hit) [N_rays]
        """
        # Vector from ray origin to sphere center
        oc = ray_origins - sphere.center.to(ray_origins.device)

        # Quadratic coefficients: at^2 + bt + c = 0
        a = (ray_directions * ray_directions).sum(dim=-1)
        b = 2.0 * (oc * ray_directions).sum(dim=-1)
        c = (oc * oc).sum(dim=-1) - sphere.radius ** 2

        discriminant = b ** 2 - 4 * a * c

        # No intersection if discriminant < 0
        hit_mask = discriminant >= 0

        # Solve for t (take smaller positive root)
        sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0))
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # Take smallest positive t
        t_hit = torch.where(t1 > 0, t1, t2)
        t_hit = torch.where(hit_mask & (t_hit > 0), t_hit, torch.full_like(t_hit, float("inf")))

        return hit_mask, t_hit

    def _render_view(self, pose: CameraPose) -> torch.Tensor:
        """
        Render ground truth image by ray tracing spheres.

        For each pixel, finds the nearest sphere intersection and
        uses simple diffuse shading.
        """
        device = self.device
        H = W = self.image_size

        # Generate rays
        ray_origins, ray_directions = generate_rays(self.intrinsics, pose, device)

        # Initialize with background color (white)
        image = torch.ones(H * W, 3, device=device)

        # Track nearest hit distance
        nearest_t = torch.full((H * W,), float("inf"), device=device)
        hit_colors = torch.ones(H * W, 3, device=device)

        # Light direction (from upper right)
        light_dir = F.normalize(torch.tensor([1.0, 1.0, 1.0], device=device), dim=0)

        for sphere in self.spheres:
            hit_mask, t_hit = self._ray_sphere_intersection(
                ray_origins, ray_directions, sphere
            )

            # Check if this hit is closer
            closer_mask = hit_mask & (t_hit < nearest_t)

            # Compute hit points and normals for closer hits
            hit_points = ray_origins + t_hit.unsqueeze(-1) * ray_directions
            normals = F.normalize(hit_points - sphere.center.to(device), dim=-1)

            # Simple diffuse shading
            diffuse = torch.clamp((normals * light_dir).sum(dim=-1), 0.1, 1.0)
            shaded_color = sphere.color.to(device) * diffuse.unsqueeze(-1)

            # Update colors where this sphere is closer
            hit_colors = torch.where(
                closer_mask.unsqueeze(-1),
                shaded_color,
                hit_colors
            )
            nearest_t = torch.where(closer_mask, t_hit, nearest_t)

        # Apply colors where we hit something
        hit_anything = nearest_t < float("inf")
        image = torch.where(hit_anything.unsqueeze(-1), hit_colors, image)

        return image.reshape(H, W, 3)

    def _render_all_views(self) -> List[torch.Tensor]:
        """Pre-render all training views."""
        images = []
        for pose in self.poses:
            img = self._render_view(pose)
            images.append(img)
        return images

    def __len__(self) -> int:
        return self.num_views

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dictionary with:
                - ray_origins: [H*W, 3]
                - ray_directions: [H*W, 3]
                - target_rgb: [H*W, 3]
                - pose_idx: scalar
        """
        pose = self.poses[idx]
        image = self.images[idx]

        ray_origins, ray_directions = generate_rays(
            self.intrinsics, pose, self.device
        )

        return {
            "ray_origins": ray_origins,
            "ray_directions": ray_directions,
            "target_rgb": image.reshape(-1, 3),
            "pose_idx": torch.tensor(idx)
        }

    def get_intrinsics(self) -> CameraIntrinsics:
        """Get camera intrinsics."""
        return self.intrinsics

    def get_pose(self, idx: int) -> CameraPose:
        """Get camera pose for a view."""
        return self.poses[idx]

    def get_spheres(self) -> List[Sphere]:
        """Get scene geometry for visualization."""
        return self.spheres


class RayBatchDataset(Dataset):
    """
    Dataset that samples random rays from multiple views.

    Instead of returning full images, this samples random rays
    from the training views. More efficient for NeRF training.

    Args:
        base_dataset: SyntheticSceneDataset or BlenderDataset
        rays_per_sample: Number of rays in each batch
    """

    def __init__(
        self,
        base_dataset: SyntheticSceneDataset,
        rays_per_sample: int = 1024
    ):
        self.base_dataset = base_dataset
        self.rays_per_sample = rays_per_sample

        # Pre-compute all rays and targets
        self.all_rays_o = []
        self.all_rays_d = []
        self.all_targets = []

        for i in range(len(base_dataset)):
            sample = base_dataset[i]
            self.all_rays_o.append(sample["ray_origins"])
            self.all_rays_d.append(sample["ray_directions"])
            self.all_targets.append(sample["target_rgb"])

        self.all_rays_o = torch.cat(self.all_rays_o, dim=0)
        self.all_rays_d = torch.cat(self.all_rays_d, dim=0)
        self.all_targets = torch.cat(self.all_targets, dim=0)

        self.n_rays = self.all_rays_o.shape[0]

    def __len__(self) -> int:
        # Number of batches in one epoch
        return self.n_rays // self.rays_per_sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of rays."""
        # Random indices
        indices = torch.randint(0, self.n_rays, (self.rays_per_sample,))

        return {
            "ray_origins": self.all_rays_o[indices],
            "ray_directions": self.all_rays_d[indices],
            "target_rgb": self.all_targets[indices]
        }


class BlenderDataset(Dataset):
    """
    Load NeRF synthetic dataset (Blender scenes).

    Expected directory structure:
        scene_name/
            transforms_train.json
            transforms_val.json
            transforms_test.json
            train/
                r_0.png
                r_1.png
                ...

    The JSON files contain camera intrinsics and per-image extrinsics.

    Args:
        root_dir: Path to scene directory
        split: "train", "val", or "test"
        image_size: Target image size (downsample if needed)
        white_background: Replace alpha with white background
        device: Torch device
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 64,
        white_background: bool = True,
        device: str = "cpu"
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.white_background = white_background
        self.device = device

        # Load transforms
        transforms_file = self.root_dir / f"transforms_{split}.json"
        if not transforms_file.exists():
            raise FileNotFoundError(f"Transforms file not found: {transforms_file}")

        with open(transforms_file, "r") as f:
            self.transforms = json.load(f)

        # Parse camera intrinsics from transforms
        self.camera_angle_x = self.transforms["camera_angle_x"]
        self.focal = 0.5 * image_size / np.tan(0.5 * self.camera_angle_x)

        self.intrinsics = CameraIntrinsics(
            width=image_size,
            height=image_size,
            focal_x=self.focal
        )

        # Load frames
        self.frames = self.transforms["frames"]

    def __len__(self) -> int:
        return len(self.frames)

    def _load_image(self, frame: dict) -> torch.Tensor:
        """Load and preprocess an image."""
        from PIL import Image
        import torchvision.transforms as T

        file_path = self.root_dir / f"{frame['file_path']}.png"
        if not file_path.exists():
            # Try without .png extension
            file_path = self.root_dir / frame["file_path"]

        img = Image.open(file_path)

        # Handle RGBA images
        if img.mode == "RGBA":
            img = np.array(img) / 255.0
            if self.white_background:
                # Composite over white
                alpha = img[..., 3:4]
                rgb = img[..., :3] * alpha + (1 - alpha)
            else:
                rgb = img[..., :3]
            img = torch.tensor(rgb, dtype=torch.float32)
        else:
            img = np.array(img.convert("RGB")) / 255.0
            img = torch.tensor(img, dtype=torch.float32)

        # Resize if needed
        if img.shape[0] != self.image_size:
            img = img.permute(2, 0, 1).unsqueeze(0)
            img = F.interpolate(img, size=(self.image_size, self.image_size), mode="bilinear")
            img = img.squeeze(0).permute(1, 2, 0)

        return img

    def _parse_pose(self, frame: dict) -> CameraPose:
        """Parse camera pose from transform matrix."""
        transform = torch.tensor(frame["transform_matrix"], dtype=torch.float32)

        # Extract rotation and translation
        rotation = transform[:3, :3]
        translation = transform[:3, 3]

        return CameraPose(rotation=rotation, translation=translation)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        frame = self.frames[idx]

        # Load image
        image = self._load_image(frame).to(self.device)

        # Parse pose
        pose = self._parse_pose(frame)

        # Generate rays
        ray_origins, ray_directions = generate_rays(
            self.intrinsics, pose, self.device
        )

        return {
            "ray_origins": ray_origins,
            "ray_directions": ray_directions,
            "target_rgb": image.reshape(-1, 3),
            "pose_idx": torch.tensor(idx)
        }

    def get_intrinsics(self) -> CameraIntrinsics:
        return self.intrinsics

    def get_pose(self, idx: int) -> CameraPose:
        return self._parse_pose(self.frames[idx])


if __name__ == "__main__":
    # Test datasets
    print("Testing datasets...")

    # Test synthetic dataset
    print("\n=== Synthetic Scene Dataset ===")
    scene_config = SyntheticScene(num_spheres=3, seed=42)
    dataset = SyntheticSceneDataset(
        scene_config=scene_config,
        num_views=10,
        image_size=32,
        device="cpu"
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Spheres: {len(dataset.get_spheres())}")

    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Ray origins shape: {sample['ray_origins'].shape}")
    print(f"Ray directions shape: {sample['ray_directions'].shape}")
    print(f"Target RGB shape: {sample['target_rgb'].shape}")

    # Check image values are in valid range
    assert sample["target_rgb"].min() >= 0 and sample["target_rgb"].max() <= 1

    # Test ray batch dataset
    print("\n=== Ray Batch Dataset ===")
    ray_dataset = RayBatchDataset(dataset, rays_per_sample=256)
    print(f"Ray dataset size: {len(ray_dataset)}")

    batch = ray_dataset[0]
    print(f"Batch ray origins shape: {batch['ray_origins'].shape}")
    assert batch["ray_origins"].shape == (256, 3)

    # Save a test image
    print("\n=== Saving test render ===")
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for i in range(3):
            img = dataset.images[i].numpy()
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"View {i}")
        plt.tight_layout()
        plt.savefig("pretrained/synthetic_test.png", dpi=100)
        plt.close()
        print("Saved test render to pretrained/synthetic_test.png")
    except ImportError:
        print("Matplotlib not available, skipping image save")

    print("\nAll dataset tests passed!")
