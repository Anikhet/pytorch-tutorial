"""
Neural Network From Scratch - No Libraries!
Shows how the racing car brain works with just basic math.
"""

import numpy as np  # Only using numpy for matrix operations, not neural networks!

class SimpleNeuralNetwork:
    """
    A neural network implemented from scratch.
    Architecture: 5 → 8 → 8 → 2 (same as the racing car)
    """

    def __init__(self):
        """Initialize weights randomly."""
        # Layer 1: 5 inputs → 8 neurons
        self.W1 = np.random.randn(8, 5) * 0.5  # 8x5 weight matrix
        self.b1 = np.zeros(8)                   # 8 biases

        # Layer 2: 8 inputs → 8 neurons
        self.W2 = np.random.randn(8, 8) * 0.5  # 8x8 weight matrix
        self.b2 = np.zeros(8)                   # 8 biases

        # Layer 3: 8 inputs → 2 outputs
        self.W3 = np.random.randn(2, 8) * 0.5  # 2x8 weight matrix
        self.b3 = np.zeros(2)                   # 2 biases

        print(f"Created network with {self.count_parameters()} parameters")

    def relu(self, x):
        """
        ReLU activation: if x < 0, return 0, else return x
        This adds non-linearity to the network.
        """
        return np.maximum(0, x)

    def tanh(self, x):
        """
        Tanh activation: squashes values to range [-1, 1]
        Formula: (e^x - e^-x) / (e^x + e^-x)
        """
        return np.tanh(x)

    def forward(self, x):
        """
        Forward pass: input → hidden1 → hidden2 → output

        This is the EXACT same math that PyTorch does internally!
        """
        # Layer 1: Linear transformation + ReLU
        z1 = np.dot(self.W1, x) + self.b1  # W*x + b
        a1 = self.relu(z1)                  # ReLU activation

        # Layer 2: Linear transformation + ReLU
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.relu(z2)

        # Layer 3: Linear transformation + Tanh
        z3 = np.dot(self.W3, a2) + self.b3
        a3 = self.tanh(z3)                  # Tanh activation

        return a3  # [acceleration, steering]

    def predict(self, sensor_data):
        """Same interface as PyTorch version."""
        return self.forward(sensor_data)

    def get_weights(self):
        """Get all weights as a flat array (for genetic algorithm)."""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten(),
            self.W3.flatten(),
            self.b3.flatten()
        ])

    def set_weights(self, weights):
        """Set weights from a flat array (for genetic algorithm)."""
        idx = 0

        # Layer 1 weights
        W1_size = self.W1.size
        self.W1 = weights[idx:idx + W1_size].reshape(self.W1.shape)
        idx += W1_size

        # Layer 1 biases
        b1_size = self.b1.size
        self.b1 = weights[idx:idx + b1_size]
        idx += b1_size

        # Layer 2 weights
        W2_size = self.W2.size
        self.W2 = weights[idx:idx + W2_size].reshape(self.W2.shape)
        idx += W2_size

        # Layer 2 biases
        b2_size = self.b2.size
        self.b2 = weights[idx:idx + b2_size]
        idx += b2_size

        # Layer 3 weights
        W3_size = self.W3.size
        self.W3 = weights[idx:idx + W3_size].reshape(self.W3.shape)
        idx += W3_size

        # Layer 3 biases
        b3_size = self.b3.size
        self.b3 = weights[idx:idx + b3_size]

    def mutate(self, mutation_rate=0.1, mutation_scale=0.3):
        """Mutate weights for genetic algorithm."""
        weights = self.get_weights()

        # Randomly mutate some weights
        mutation_mask = np.random.random(len(weights)) < mutation_rate
        mutations = np.random.randn(len(weights)) * mutation_scale
        weights[mutation_mask] += mutations[mutation_mask]

        self.set_weights(weights)

    def copy(self):
        """Create a copy of this network."""
        new_network = SimpleNeuralNetwork()
        new_network.set_weights(self.get_weights())
        return new_network

    def count_parameters(self):
        """Count total number of parameters."""
        return self.W1.size + self.b1.size + \
               self.W2.size + self.b2.size + \
               self.W3.size + self.b3.size


# ============ DEMO: Compare with PyTorch ============

if __name__ == "__main__":
    print("="*60)
    print("NEURAL NETWORK FROM SCRATCH - DEMO")
    print("="*60)

    # Create our from-scratch network
    print("\n1. Creating network from scratch...")
    scratch_net = SimpleNeuralNetwork()

    # Test with sample sensor data
    sensor_data = np.array([0.3, 0.5, 0.7, 0.9, 0.95])

    print("\n2. Running forward pass...")
    print(f"   Input (sensors): {sensor_data}")

    # Forward pass
    output = scratch_net.forward(sensor_data)

    print(f"\n3. Output:")
    print(f"   Acceleration: {output[0]:+.3f}")
    print(f"   Steering:     {output[1]:+.3f}")

    # Show what's happening inside
    print("\n4. Internal computations:")

    # Layer 1 in detail
    print("\n   Layer 1 (5 → 8):")
    z1 = np.dot(scratch_net.W1, sensor_data) + scratch_net.b1
    a1 = scratch_net.relu(z1)
    print(f"   Before ReLU: {z1[:3]} ... (showing first 3)")
    print(f"   After ReLU:  {a1[:3]} ... (showing first 3)")

    # Layer 2 in detail
    print("\n   Layer 2 (8 → 8):")
    z2 = np.dot(scratch_net.W2, a1) + scratch_net.b2
    a2 = scratch_net.relu(z2)
    print(f"   Before ReLU: {z2[:3]} ... (showing first 3)")
    print(f"   After ReLU:  {a2[:3]} ... (showing first 3)")

    # Layer 3 in detail
    print("\n   Layer 3 (8 → 2):")
    z3 = np.dot(scratch_net.W3, a2) + scratch_net.b3
    a3 = scratch_net.tanh(z3)
    print(f"   Before Tanh: {z3}")
    print(f"   After Tanh:  {a3}")

    # Test genetic algorithm operations
    print("\n5. Testing genetic algorithm operations:")

    # Get weights
    weights = scratch_net.get_weights()
    print(f"   Total weights: {len(weights)}")
    print(f"   First 5 weights: {weights[:5]}")

    # Mutate
    print("\n   Mutating network...")
    scratch_net.mutate(mutation_rate=0.2)
    new_weights = scratch_net.get_weights()
    print(f"   Changed weights: {np.sum(weights != new_weights)}")

    # Copy
    print("\n   Creating a copy...")
    copy_net = scratch_net.copy()
    print(f"   Copy produces same output: {np.allclose(copy_net.forward(sensor_data), scratch_net.forward(sensor_data))}")

    print("\n" + "="*60)
    print("✅ This is ALL a neural network is - just math!")
    print("PyTorch does the same thing, but with:")
    print("  • Automatic GPU acceleration")
    print("  • Backpropagation (gradient calculation)")
    print("  • Optimized C++ code")
    print("  • Lots of convenience features")
    print("\nFor this simple network, we don't need any of that! 🎉")
    print("="*60)
