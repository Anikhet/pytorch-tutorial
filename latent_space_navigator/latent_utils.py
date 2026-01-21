"""
Latent space utilities for semantic direction finding and interpolation.

Features:
- Encode full dataset to compute class statistics
- Find semantic directions (digit identity, thickness, slant)
- Linear and spherical interpolation
- Latent space visualization helpers
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


@torch.no_grad()
def encode_dataset(
    model,
    dataloader: DataLoader,
    device: str = "cpu",
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode all images in dataset to latent space.

    Args:
        model: VAE model with encode method
        dataloader: DataLoader for the dataset
        device: Device to run on
        max_samples: Maximum samples to encode (None for all)

    Returns:
        latent_codes: [N, latent_dim] numpy array
        labels: [N] numpy array of digit labels
    """
    model.eval()
    all_codes = []
    all_labels = []
    count = 0

    for images, labels in dataloader:
        if max_samples and count >= max_samples:
            break

        images = images.to(device)
        mu, _ = model.encode(images)
        all_codes.append(mu.cpu().numpy())
        all_labels.append(labels.numpy())
        count += len(labels)

    latent_codes = np.concatenate(all_codes, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if max_samples:
        latent_codes = latent_codes[:max_samples]
        labels = labels[:max_samples]

    return latent_codes, labels


def compute_class_centroids(
    latent_codes: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 10
) -> Dict[int, np.ndarray]:
    """
    Compute mean latent code for each digit class.

    Args:
        latent_codes: [N, latent_dim] encoded samples
        labels: [N] digit labels

    Returns:
        Dictionary mapping digit -> centroid vector
    """
    centroids = {}
    for digit in range(num_classes):
        mask = labels == digit
        if mask.sum() > 0:
            centroids[digit] = latent_codes[mask].mean(axis=0)
    return centroids


def find_digit_directions(
    centroids: Dict[int, np.ndarray]
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Compute direction vectors between digit centroids.

    Args:
        centroids: Dictionary of digit centroids

    Returns:
        Dictionary mapping (from_digit, to_digit) -> direction vector
    """
    directions = {}
    digits = sorted(centroids.keys())

    for i, from_digit in enumerate(digits):
        for to_digit in digits[i + 1:]:
            direction = centroids[to_digit] - centroids[from_digit]
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            directions[(from_digit, to_digit)] = direction
            directions[(to_digit, from_digit)] = -direction

    return directions


def find_semantic_directions_pca(
    latent_codes: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2
) -> Tuple[np.ndarray, PCA]:
    """
    Find principal variation directions across classes using PCA on centroids.

    Args:
        latent_codes: [N, latent_dim] encoded samples
        labels: [N] digit labels
        n_components: Number of principal components

    Returns:
        directions: [n_components, latent_dim] principal directions
        pca: Fitted PCA object
    """
    centroids = compute_class_centroids(latent_codes, labels)
    centroid_matrix = np.stack([centroids[d] for d in sorted(centroids.keys())])

    pca = PCA(n_components=min(n_components, len(centroids) - 1))
    pca.fit(centroid_matrix)

    return pca.components_, pca


def find_semantic_directions_classifier(
    latent_codes: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    Train a linear classifier to find digit directions.

    The weight vectors point toward each digit class.

    Args:
        latent_codes: [N, latent_dim] encoded samples
        labels: [N] digit labels

    Returns:
        directions: [10, latent_dim] direction for each digit
    """
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    clf.fit(latent_codes, labels)
    return clf.coef_


def linear_interpolation(
    z1: np.ndarray,
    z2: np.ndarray,
    num_steps: int = 10
) -> np.ndarray:
    """
    Linear interpolation between two latent points.

    Args:
        z1: Start point [latent_dim]
        z2: End point [latent_dim]
        num_steps: Number of interpolation steps

    Returns:
        points: [num_steps, latent_dim] interpolation path
    """
    t = np.linspace(0, 1, num_steps)
    return np.array([(1 - ti) * z1 + ti * z2 for ti in t])


def spherical_interpolation(
    z1: np.ndarray,
    z2: np.ndarray,
    num_steps: int = 10
) -> np.ndarray:
    """
    Spherical linear interpolation (slerp) between two latent points.

    Better for traversing latent space as it follows geodesics.

    Args:
        z1: Start point [latent_dim]
        z2: End point [latent_dim]
        num_steps: Number of interpolation steps

    Returns:
        points: [num_steps, latent_dim] interpolation path
    """
    z1_norm = z1 / (np.linalg.norm(z1) + 1e-8)
    z2_norm = z2 / (np.linalg.norm(z2) + 1e-8)

    omega = np.arccos(np.clip(np.dot(z1_norm, z2_norm), -1, 1))

    if np.abs(omega) < 1e-6:
        return linear_interpolation(z1, z2, num_steps)

    t = np.linspace(0, 1, num_steps)
    sin_omega = np.sin(omega)

    r1 = np.linalg.norm(z1)
    r2 = np.linalg.norm(z2)

    points = []
    for ti in t:
        r_interp = (1 - ti) * r1 + ti * r2
        direction = (np.sin((1 - ti) * omega) * z1_norm +
                    np.sin(ti * omega) * z2_norm) / sin_omega
        points.append(r_interp * direction)

    return np.array(points)


def move_along_direction(
    z: np.ndarray,
    direction: np.ndarray,
    amount: float
) -> np.ndarray:
    """
    Move latent point along a semantic direction.

    Args:
        z: Current latent point [latent_dim]
        direction: Direction vector [latent_dim]
        amount: How far to move (can be negative)

    Returns:
        New latent point
    """
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
    return z + amount * direction_norm


class LatentSpaceAnalyzer:
    """
    Encapsulates latent space analysis and semantic directions.

    Usage:
        analyzer = LatentSpaceAnalyzer(model, dataloader)
        analyzer.analyze()

        new_z = analyzer.move_toward_digit(current_z, target_digit=5, amount=1.0)
    """

    def __init__(
        self,
        model,
        dataloader: DataLoader,
        device: str = "cpu",
        max_samples: int = 10000
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.max_samples = max_samples

        self.latent_codes = None
        self.labels = None
        self.centroids = None
        self.digit_directions = None
        self.pca_directions = None

    def analyze(self):
        """Run full analysis of latent space."""
        print("Encoding dataset...")
        self.latent_codes, self.labels = encode_dataset(
            self.model, self.dataloader, self.device, self.max_samples
        )

        print("Computing class centroids...")
        self.centroids = compute_class_centroids(self.latent_codes, self.labels)

        print("Finding digit directions...")
        self.digit_directions = find_digit_directions(self.centroids)

        print("Computing PCA directions...")
        self.pca_directions, _ = find_semantic_directions_pca(
            self.latent_codes, self.labels
        )

        print("Analysis complete!")

    def get_centroid(self, digit: int) -> np.ndarray:
        """Get centroid for a specific digit."""
        if self.centroids is None:
            raise ValueError("Run analyze() first")
        return self.centroids[digit]

    def move_toward_digit(
        self,
        z: np.ndarray,
        target_digit: int,
        amount: float = 1.0
    ) -> np.ndarray:
        """
        Move latent point toward a target digit's centroid.

        Args:
            z: Current latent point
            target_digit: Digit to move toward (0-9)
            amount: Interpolation amount (0=stay, 1=reach centroid)

        Returns:
            New latent point
        """
        if self.centroids is None:
            raise ValueError("Run analyze() first")

        target = self.centroids[target_digit]
        return (1 - amount) * z + amount * target

    def interpolate_digits(
        self,
        from_digit: int,
        to_digit: int,
        num_steps: int = 10,
        method: str = "linear"
    ) -> np.ndarray:
        """
        Interpolate between two digit centroids.

        Args:
            from_digit: Starting digit
            to_digit: Ending digit
            num_steps: Number of interpolation steps
            method: "linear" or "spherical"

        Returns:
            [num_steps, latent_dim] interpolation path
        """
        if self.centroids is None:
            raise ValueError("Run analyze() first")

        z1 = self.centroids[from_digit]
        z2 = self.centroids[to_digit]

        if method == "spherical":
            return spherical_interpolation(z1, z2, num_steps)
        return linear_interpolation(z1, z2, num_steps)

    def get_latent_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounds of the latent space for visualization."""
        if self.latent_codes is None:
            raise ValueError("Run analyze() first")

        x_min, x_max = self.latent_codes[:, 0].min(), self.latent_codes[:, 0].max()
        y_min, y_max = self.latent_codes[:, 1].min(), self.latent_codes[:, 1].max()

        margin = 0.5
        return x_min - margin, x_max + margin, y_min - margin, y_max + margin


if __name__ == "__main__":
    print("Testing latent utilities...")

    z1 = np.array([0.0, 0.0])
    z2 = np.array([1.0, 1.0])

    linear_path = linear_interpolation(z1, z2, 5)
    print(f"Linear interpolation: {linear_path.shape}")

    slerp_path = spherical_interpolation(z1, z2, 5)
    print(f"Spherical interpolation: {slerp_path.shape}")

    direction = np.array([1.0, 0.0])
    moved = move_along_direction(z1, direction, 0.5)
    print(f"Moved point: {moved}")

    fake_codes = np.random.randn(100, 2)
    fake_labels = np.random.randint(0, 10, 100)

    centroids = compute_class_centroids(fake_codes, fake_labels)
    print(f"Computed {len(centroids)} centroids")

    directions = find_digit_directions(centroids)
    print(f"Found {len(directions)} direction pairs")

    print("All tests passed!")
