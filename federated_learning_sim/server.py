"""
Federated learning server implementing FedAvg.

The server orchestrates the federated learning process:
  1. Initialize a global model
  2. Each round: send weights to clients, collect updates, average them
  3. Test the averaged model on a held-out test set
  4. Repeat for a fixed number of rounds

The core algorithm is FedAvg (Federated Averaging) from McMahan et al., 2017.
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SimpleCNN
from data import get_mnist, split_iid, split_non_iid
from client import FederatedClient, train_client_with_dp


def federated_average(client_weights_list):
    """
    Average model weights from multiple clients (FedAvg).

    Each client contributes equally to the global model. This is the
    simplest aggregation strategy -- more advanced methods weight by
    dataset size or use secure aggregation.

    Args:
        client_weights_list: List of state_dicts from each client.

    Returns:
        A new state_dict with averaged parameters.
    """
    # Start from a deep copy of the first client's weights
    averaged = copy.deepcopy(client_weights_list[0])
    num_clients = len(client_weights_list)

    for key in averaged:
        # Sum all client weights for this parameter
        for i in range(1, num_clients):
            averaged[key] = averaged[key] + client_weights_list[i][key]
        # Divide by number of clients
        averaged[key] = torch.div(averaged[key], num_clients)

    return averaged


def test_model(model, test_loader):
    """
    Measure model accuracy on a test set.

    Args:
        model:       The model to test.
        test_loader: DataLoader for the test set.

    Returns:
        Accuracy as a float between 0 and 100.
    """
    model.train(False)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def run_simulation(num_clients=5, num_rounds=10, local_epochs=3,
                   lr=0.01, iid=True, use_dp=False,
                   noise_multiplier=1.0, max_grad_norm=1.0):
    """
    Run a complete federated learning simulation.

    Args:
        num_clients:      Number of federated clients.
        num_rounds:       Number of communication rounds.
        local_epochs:     Training epochs per client per round.
        lr:               Learning rate for local SGD.
        iid:              If True, use IID data split; else non-IID.
        use_dp:           If True, use DP-SGD for local training.
        noise_multiplier: DP noise scale (only used if use_dp=True).
        max_grad_norm:    DP gradient clip norm (only used if use_dp=True).

    Returns:
        Dict mapping round number (1-indexed) to test accuracy.
    """
    # Load data and create client partitions
    train_dataset, test_dataset = get_mnist()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    if iid:
        client_loaders = split_iid(train_dataset, num_clients)
    else:
        client_loaders = split_non_iid(train_dataset, num_clients)

    # Create clients and global model
    clients = [
        FederatedClient(client_id=i, train_loader=client_loaders[i])
        for i in range(num_clients)
    ]
    global_model = SimpleCNN()

    history = {}

    for round_num in range(1, num_rounds + 1):
        # Collect updated weights from all clients
        client_weights = []
        for client in clients:
            if use_dp:
                weights = train_client_with_dp(
                    client, global_model, local_epochs, lr,
                    noise_multiplier, max_grad_norm,
                )
            else:
                weights = client.train(global_model, local_epochs, lr)
            client_weights.append(weights)

        # Aggregate via FedAvg
        averaged_weights = federated_average(client_weights)
        global_model.load_state_dict(averaged_weights)

        # Check accuracy on test set
        accuracy = test_model(global_model, test_loader)
        history[round_num] = accuracy

        dp_tag = " [DP]" if use_dp else ""
        print(f"Round {round_num:2d}/{num_rounds} - Accuracy: {accuracy:.2f}%{dp_tag}")

    return history


if __name__ == "__main__":
    print("=== Federated Learning Simulation ===")
    print("Clients: 5 | Rounds: 10 | IID: True\n")
    run_simulation(num_clients=5, num_rounds=10, iid=True)
