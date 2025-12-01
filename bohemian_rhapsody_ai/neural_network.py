"""
Neural network for controlling the guitar player.
"""

import torch
import torch.nn as nn
import numpy as np
import copy


class GuitarNetwork(nn.Module):
    """
    Neural network that decides which string/fret to play and when.

    Input: 8 sensor values
    Hidden: 12 neurons, 12 neurons
    Output: 3 actions (string, fret, play trigger)
    """

    def __init__(self):
        super(GuitarNetwork, self).__init__()

        # Network architecture
        self.fc1 = nn.Linear(8, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 3)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of sensor data (8 values)

        Returns:
            Output tensor (3 values): string, fret, play_trigger
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))  # Outputs in range [-1, 1]
        return x

    def predict(self, sensor_data):
        """
        Make a prediction given sensor data.

        Args:
            sensor_data: Numpy array of 8 sensor values

        Returns:
            Numpy array of 3 action values
        """
        with torch.no_grad():
            x = torch.FloatTensor(sensor_data)
            output = self.forward(x)
            return output.numpy()

    def copy(self):
        """Create a deep copy of this network."""
        new_network = GuitarNetwork()
        new_network.load_state_dict(copy.deepcopy(self.state_dict()))
        return new_network

    def get_weights(self):
        """
        Get all network weights as a flat numpy array.
        Used for genetic algorithm operations.
        """
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def set_weights(self, weights):
        """
        Set network weights from a flat numpy array.
        Used for genetic algorithm operations.
        """
        idx = 0
        for param in self.parameters():
            param_shape = param.data.shape
            param_size = param.data.numel()
            param_weights = weights[idx:idx + param_size]
            param.data = torch.FloatTensor(param_weights.reshape(param_shape))
            idx += param_size

    def mutate(self, mutation_rate=0.1, mutation_scale=0.3):
        """
        Randomly mutate network weights.

        Args:
            mutation_rate: Probability of mutating each weight
            mutation_scale: Scale of random mutation
        """
        for param in self.parameters():
            if np.random.random() < mutation_rate:
                # Add random noise to this parameter
                noise = torch.randn_like(param.data) * mutation_scale
                param.data += noise

    @staticmethod
    def crossover(parent1, parent2):
        """
        Create a child network by crossing over two parent networks.

        Args:
            parent1: First parent network
            parent2: Second parent network

        Returns:
            New child network
        """
        child = GuitarNetwork()

        # Get weights from both parents
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()

        # Single-point crossover
        crossover_point = np.random.randint(0, len(weights1))
        child_weights = np.concatenate([
            weights1[:crossover_point],
            weights2[crossover_point:]
        ])

        # Set child weights
        child.set_weights(child_weights)

        return child

    def save(self, filepath):
        """Save network weights to file."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """Load network weights from file."""
        self.load_state_dict(torch.load(filepath))

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


def create_random_network():
    """Create a new network with random weights."""
    return GuitarNetwork()


if __name__ == "__main__":
    # Test the network
    print("Testing GuitarNetwork...")

    net = GuitarNetwork()
    print(f"Network created with {net.count_parameters()} parameters")

    # Test forward pass
    test_input = np.random.rand(8)
    output = net.predict(test_input)
    print(f"\nTest input: {test_input}")
    print(f"Test output: {output}")

    # Test mutation
    print("\n\nTesting mutation...")
    weights_before = net.get_weights().copy()
    net.mutate(mutation_rate=0.5, mutation_scale=0.1)
    weights_after = net.get_weights()
    mutation_diff = np.abs(weights_after - weights_before).mean()
    print(f"Average weight change: {mutation_diff:.6f}")

    # Test crossover
    print("\nTesting crossover...")
    parent1 = create_random_network()
    parent2 = create_random_network()
    child = GuitarNetwork.crossover(parent1, parent2)
    print(f"Child network created from crossover")

    # Test copy
    print("\nTesting copy...")
    copy_net = net.copy()
    print(f"Network copied successfully")

    print("\nAll tests passed!")
