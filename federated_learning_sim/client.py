"""
Federated learning client.

Each client holds a private dataset and trains a local copy of the
global model. Only model weights (never raw data) are sent back
to the server. This is the core privacy benefit of federated learning.

The DP-SGD training variant is kept as a separate function to
maintain clean separation between standard training and privacy logic.
"""

import copy

import torch
import torch.nn as nn

from privacy import clip_gradients, add_noise


class FederatedClient:
    """
    A federated learning client that trains locally on private data.

    Each client:
      1. Receives the global model weights from the server
      2. Creates a local copy (never mutates the global model)
      3. Trains on its private data for a few epochs
      4. Returns the updated weights to the server
    """

    def __init__(self, client_id, train_loader):
        """
        Args:
            client_id:    Unique identifier for this client.
            train_loader: DataLoader with this client's private data.
        """
        self.client_id = client_id
        self.train_loader = train_loader

    def train(self, global_model, epochs=3, lr=0.01):
        """
        Train a local copy of the global model on private data.

        IMPORTANT: We deep-copy the global model so the original
        is never mutated. This follows immutability principles and
        mirrors real FL where the server model stays untouched.

        Args:
            global_model: The current global model (not modified).
            epochs:       Number of local training epochs.
            lr:           Learning rate for SGD optimizer.

        Returns:
            New state_dict with updated weights after local training.
        """
        # Create an independent local copy
        local_model = copy.deepcopy(global_model)
        local_model.train()

        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _epoch in range(epochs):
            for images, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Return a detached copy of the state dict
        return copy.deepcopy(local_model.state_dict())


def train_client_with_dp(client, global_model, epochs=3, lr=0.01,
                         noise_multiplier=1.0, max_grad_norm=1.0):
    """
    Train a client's local model with differential privacy (DP-SGD).

    DP-SGD modifies standard SGD in two ways:
      1. Gradient clipping: Caps the L2 norm of each gradient update,
         bounding any single sample's influence.
      2. Noise addition: Adds Gaussian noise calibrated to the clip
         norm, masking individual contributions.

    This function is intentionally separate from FederatedClient.train()
    to keep privacy logic decoupled from standard training.

    Args:
        client:           FederatedClient instance with private data.
        global_model:     The current global model (not modified).
        epochs:           Number of local training epochs.
        lr:               Learning rate for SGD optimizer.
        noise_multiplier: DP noise scale (higher = more private).
        max_grad_norm:    Maximum L2 norm for gradient clipping.

    Returns:
        New state_dict with DP-trained weights.
    """
    # Create an independent local copy -- never mutate the original
    local_model = copy.deepcopy(global_model)
    local_model.train()

    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _epoch in range(epochs):
        for images, labels in client.train_loader:
            optimizer.zero_grad()
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # DP-SGD: clip gradients then add noise
            clip_gradients(local_model, max_grad_norm)
            add_noise(local_model, noise_multiplier, max_grad_norm)

            optimizer.step()

    return copy.deepcopy(local_model.state_dict())
