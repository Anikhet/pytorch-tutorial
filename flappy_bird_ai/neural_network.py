import numpy as np
import torch
import torch.nn as nn

class BirdNeuralNetwork(nn.Module):
    """Simple neural network for Flappy Bird decision making."""

    def __init__(self, input_size=4, hidden_size=8, output_size=1):
        """
        Args:
            input_size: Number of inputs (bird y, distance to pipe, gap y, velocity)
            hidden_size: Number of neurons in hidden layer
            output_size: Number of outputs (jump or not)
        """
        super(BirdNeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Output 0-1 (jump threshold)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Forward pass through the network."""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        return self.network(x)

    def predict(self, inputs):
        """
        Predict action from inputs.

        Args:
            inputs: numpy array of sensor readings

        Returns:
            output: [jump_probability] (0-1)
        """
        with torch.no_grad():
            output = self.forward(inputs)
            return output.numpy()

    def get_weights(self):
        """Get all network weights as a flat numpy array."""
        weights = []
        for param in self.parameters():
            weights.extend(param.data.cpu().numpy().flatten())
        return np.array(weights)

    def set_weights(self, weights):
        """Set network weights from a flat numpy array."""
        idx = 0
        for param in self.parameters():
            param_shape = param.shape
            param_size = param.numel()
            param.data = torch.FloatTensor(
                weights[idx:idx + param_size].reshape(param_shape)
            )
            idx += param_size

    def get_num_weights(self):
        """Get total number of weights in the network."""
        return sum(p.numel() for p in self.parameters())

    def copy(self):
        """Create a deep copy of the network."""
        new_network = BirdNeuralNetwork(
            input_size=self.network[0].in_features,
            hidden_size=self.network[0].out_features,
            output_size=self.network[-2].out_features
        )
        new_network.set_weights(self.get_weights())
        return new_network

    def mutate(self, mutation_rate=0.1, mutation_scale=0.3):
        """
        Mutate network weights for genetic algorithm.

        Args:
            mutation_rate: Probability of mutating each weight
            mutation_scale: Scale of random mutations
        """
        weights = self.get_weights()

        # Randomly mutate some weights
        mutation_mask = np.random.random(len(weights)) < mutation_rate
        mutations = np.random.randn(len(weights)) * mutation_scale
        weights[mutation_mask] += mutations[mutation_mask]

        self.set_weights(weights)

    @staticmethod
    def crossover(parent1, parent2):
        """
        Create a child network by crossing over two parent networks.

        Args:
            parent1: First parent network
            parent2: Second parent network

        Returns:
            child: New network with mixed weights
        """
        child = parent1.copy()

        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()

        # Uniform crossover
        mask = np.random.random(len(weights1)) < 0.5
        child_weights = np.where(mask, weights1, weights2)

        child.set_weights(child_weights)
        return child
