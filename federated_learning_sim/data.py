"""
Data loading and partitioning for federated learning.

Provides two splitting strategies:
  - IID:     Each client gets a random, equal-sized subset of the data.
  - Non-IID: Each client gets data from only a few digit classes,
             simulating realistic heterogeneous data distributions.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist(data_dir="./data"):
    """
    Download and return MNIST train and test datasets.

    Args:
        data_dir: Directory to store/load the MNIST data.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


def split_iid(dataset, num_clients, batch_size=32):
    """
    Split dataset into equal-sized IID partitions for each client.

    Each client receives a random subset of approximately
    len(dataset) / num_clients samples drawn uniformly.

    Args:
        dataset:     The full training dataset.
        num_clients: Number of federated clients.
        batch_size:  Batch size for each client's DataLoader.

    Returns:
        List of DataLoaders, one per client.
    """
    # Shuffle indices and split into equal chunks
    total = len(dataset)
    indices = torch.randperm(total).tolist()
    shard_size = total // num_clients

    client_loaders = []
    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size
        client_indices = indices[start:end]
        subset = Subset(dataset, client_indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    return client_loaders


def split_non_iid(dataset, num_clients, classes_per_client=2, batch_size=32):
    """
    Split dataset into non-IID partitions where each client gets
    only a subset of digit classes.

    Strategy:
      1. Group all sample indices by their label (0-9).
      2. Assign each client a set of `classes_per_client` classes
         using round-robin allocation.
      3. Evenly divide samples of each assigned class among clients
         that share that class.

    This creates realistic heterogeneity -- e.g., Client 0 might
    only see digits 0 and 1, while Client 1 sees digits 2 and 3.

    Args:
        dataset:           The full training dataset.
        num_clients:       Number of federated clients.
        classes_per_client: How many digit classes each client receives.
        batch_size:        Batch size for each client's DataLoader.

    Returns:
        List of DataLoaders, one per client.
    """
    # Group indices by label
    label_indices = {i: [] for i in range(10)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_indices[label].append(idx)

    # Assign classes to clients via round-robin
    # Each client gets `classes_per_client` consecutive classes
    client_indices = {i: [] for i in range(num_clients)}
    for client_id in range(num_clients):
        assigned_classes = [
            (client_id * classes_per_client + j) % 10
            for j in range(classes_per_client)
        ]
        for cls in assigned_classes:
            # Give this client a portion of the class's samples
            cls_indices = label_indices[cls]
            # Count how many clients share this class
            sharing_clients = sum(
                1 for c in range(num_clients)
                if cls in [
                    (c * classes_per_client + j) % 10
                    for j in range(classes_per_client)
                ]
            )
            shard_size = len(cls_indices) // max(sharing_clients, 1)
            # Determine which shard this client gets
            shard_id = sum(
                1 for c in range(client_id)
                if cls in [
                    (c * classes_per_client + j) % 10
                    for j in range(classes_per_client)
                ]
            )
            start = shard_id * shard_size
            end = start + shard_size
            client_indices[client_id].extend(cls_indices[start:end])

    client_loaders = []
    for i in range(num_clients):
        subset = Subset(dataset, client_indices[i])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    return client_loaders
