"""
Dataset utilities for sketch-to-image diffusion.

Provides:
1. Dataset download helper for edges2shoes
2. SketchImageDataset for loading paired edge/image data
3. Data augmentation and preprocessing utilities
"""

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


DATASET_URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz"
DATASET_NAME = "edges2shoes"


def download_edges2shoes(data_dir: str = "data") -> Path:
    """
    Download and extract the edges2shoes dataset.

    The dataset contains paired edge maps and shoe images,
    perfect for training sketch-to-image models.

    Args:
        data_dir: Directory to store the dataset

    Returns:
        Path to the extracted dataset
    """
    data_path = Path(data_dir)
    dataset_path = data_path / DATASET_NAME

    if dataset_path.exists():
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path

    data_path.mkdir(parents=True, exist_ok=True)
    tar_path = data_path / f"{DATASET_NAME}.tar.gz"

    # Download
    if not tar_path.exists():
        print(f"Downloading {DATASET_NAME} dataset...")
        print(f"URL: {DATASET_URL}")
        print("This may take a few minutes...")

        def progress_hook(count, block_size, total_size):
            percent = count * block_size * 100 // total_size
            print(f"\rProgress: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(DATASET_URL, tar_path, progress_hook)
        print("\nDownload complete!")

    # Extract
    print(f"Extracting to {data_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_path)

    # Clean up tar file
    tar_path.unlink()
    print("Extraction complete!")

    return dataset_path


class SketchImageDataset(Dataset):
    """
    Dataset for paired sketch/edge and target images.

    The edges2shoes dataset has images where the left half is the edge map
    and the right half is the corresponding shoe image.

    Args:
        root_dir: Path to the dataset directory
        split: 'train' or 'val'
        image_size: Size to resize images to
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 32,
        augment: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        # Find image directory
        self.image_dir = self.root_dir / split
        if not self.image_dir.exists():
            raise ValueError(f"Split directory not found: {self.image_dir}")

        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"Found {len(self.image_files)} images in {split} split")

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess an image pair.

        Returns:
            Tuple of (sketch_tensor, target_tensor), both [3, H, W] in [-1, 1]
        """
        img_path = self.image_files[idx]

        # Load the combined image
        combined = Image.open(img_path).convert("RGB")
        width, height = combined.size

        # Split into edge (left) and target (right)
        edge = combined.crop((0, 0, width // 2, height))
        target = combined.crop((width // 2, 0, width, height))

        # Apply augmentation (horizontal flip)
        if self.augment and torch.rand(1).item() > 0.5:
            edge = edge.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Transform to tensors
        edge_tensor = self.transform(edge)
        target_tensor = self.transform(target)

        return edge_tensor, target_tensor


class SyntheticSketchDataset(Dataset):
    """
    Synthetic dataset for testing without downloading edges2shoes.

    Creates random "sketch-like" patterns and corresponding "images"
    using simple geometric shapes. Useful for quick testing.

    Args:
        num_samples: Number of synthetic samples to generate
        image_size: Size of generated images
    """

    def __init__(self, num_samples: int = 1000, image_size: int = 32):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a synthetic sketch-image pair."""
        import numpy as np

        # Set seed for reproducibility
        rng = np.random.RandomState(idx)

        # Create a simple shape
        img = np.ones((self.image_size, self.image_size, 3), dtype=np.float32)

        # Add random circles/rectangles
        num_shapes = rng.randint(1, 4)
        for _ in range(num_shapes):
            x, y = rng.randint(5, self.image_size - 5, 2)
            size = rng.randint(3, 8)
            color = rng.rand(3) * 0.8

            # Draw filled shape on target
            x1, y1 = max(0, x - size), max(0, y - size)
            x2, y2 = min(self.image_size, x + size), min(self.image_size, y + size)
            img[y1:y2, x1:x2] = color

        # Create edge version (simple sobel-like edge detection)
        edge = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        for c in range(3):
            dx = np.abs(np.diff(img[:, :, c], axis=1, prepend=img[:, :1, c]))
            dy = np.abs(np.diff(img[:, :, c], axis=0, prepend=img[:1, :, c]))
            edge[:, :, c] = np.clip(dx + dy, 0, 1)

        # Threshold to get clean edges
        edge = (edge.mean(axis=2, keepdims=True) > 0.1).astype(np.float32)
        edge = 1 - edge  # Invert: black lines on white

        # Convert to tensors and normalize to [-1, 1]
        edge_tensor = torch.from_numpy(edge.transpose(2, 0, 1).repeat(3, axis=0)[:3])
        target_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        edge_tensor = edge_tensor * 2 - 1
        target_tensor = target_tensor * 2 - 1

        return edge_tensor, target_tensor


# Test when run directly
if __name__ == "__main__":
    print("Testing dataset utilities...")

    # Test synthetic dataset
    print("\n1. Testing SyntheticSketchDataset...")
    synthetic = SyntheticSketchDataset(num_samples=100, image_size=32)
    edge, target = synthetic[0]
    print(f"   Edge shape: {edge.shape}, range: [{edge.min():.2f}, {edge.max():.2f}]")
    print(f"   Target shape: {target.shape}, range: [{target.min():.2f}, {target.max():.2f}]")

    # Test real dataset if it exists
    data_path = Path("data/edges2shoes")
    if data_path.exists():
        print("\n2. Testing SketchImageDataset...")
        dataset = SketchImageDataset(str(data_path), split="train", image_size=32)
        edge, target = dataset[0]
        print(f"   Edge shape: {edge.shape}")
        print(f"   Target shape: {target.shape}")
    else:
        print("\n2. edges2shoes not found. Run download_edges2shoes() to download.")

    print("\nAll tests passed!")
