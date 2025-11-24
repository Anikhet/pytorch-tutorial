"""
Interactive Neural Network Visualization
Shows exactly what happens inside the racing car's brain!
"""

import sys
sys.path.append('racing_car_rl')

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from neural_network import CarNeuralNetwork

# Create a network with the same architecture as the racing car
network = CarNeuralNetwork(input_size=5, hidden_size=8, output_size=2)

# Example sensor data (simulating a car approaching a right turn)
sensor_data = np.array([0.3, 0.5, 0.7, 0.9, 0.95])
print("="*80)
print("NEURAL NETWORK VISUALIZATION - Step by Step")
print("="*80)
print("\n📍 SCENARIO: Car approaching a right turn")
print("   - Left sensors detect walls getting closer (0.3, 0.5)")
print("   - Center sensor shows more space ahead (0.7)")
print("   - Right sensors show clear path (0.9, 0.95)")
print("\n" + "="*80)

# Get network layers
layer1 = network.network[0]  # First linear layer (5 -> 8)
layer2 = network.network[2]  # Second linear layer (8 -> 8)
layer3 = network.network[4]  # Output layer (8 -> 2)

# Convert input to tensor
input_tensor = torch.FloatTensor(sensor_data)

print("\n" + "🔵 INPUT LAYER (5 sensors)".center(80))
print("="*80)
for i, val in enumerate(sensor_data):
    angle = [-60, -30, 0, 30, 60][i]
    bar = "█" * int(val * 20)
    print(f"  Sensor {i+1} ({angle:+3d}°): {val:.2f} {bar}")

# LAYER 1: Input -> Hidden1
print("\n" + "🟢 HIDDEN LAYER 1 (8 neurons)".center(80))
print("="*80)
print("\nEach neuron multiplies inputs by weights and adds a bias:\n")

# Compute first layer manually to show the math
hidden1_pre_activation = layer1(input_tensor).detach().numpy()
hidden1_post_activation = np.maximum(0, hidden1_pre_activation)  # ReLU

# Show detailed calculation for first 2 neurons
for neuron_idx in range(2):
    print(f"\n  Neuron {neuron_idx + 1} calculation:")
    weights = layer1.weight[neuron_idx].detach().numpy()
    bias = layer1.bias[neuron_idx].item()

    calculation = ""
    total = bias
    for i, (sensor, weight) in enumerate(zip(sensor_data, weights)):
        calculation += f"({sensor:.2f} × {weight:+.2f}) + "
        total += sensor * weight

    calculation += f"bias({bias:+.2f})"
    print(f"    = {calculation}")
    print(f"    = {total:.3f}")
    print(f"    After ReLU: {max(0, total):.3f}")

print(f"\n  ... (showing 2 of 8 neurons)\n")
print("  All Hidden Layer 1 activations:")
for i, val in enumerate(hidden1_post_activation):
    bar = "█" * int(min(val * 3, 40))
    print(f"    Neuron {i+1}: {val:6.3f} {bar}")

# LAYER 2: Hidden1 -> Hidden2
print("\n" + "🟢 HIDDEN LAYER 2 (8 neurons)".center(80))
print("="*80)
print("\nEach neuron processes the 8 outputs from Hidden Layer 1:\n")

hidden1_tensor = torch.FloatTensor(hidden1_post_activation)
hidden2_pre_activation = layer2(hidden1_tensor).detach().numpy()
hidden2_post_activation = np.maximum(0, hidden2_pre_activation)  # ReLU

# Show detailed calculation for first neuron
neuron_idx = 0
print(f"  Neuron {neuron_idx + 1} calculation:")
weights = layer2.weight[neuron_idx].detach().numpy()
bias = layer2.bias[neuron_idx].item()

calculation_parts = []
total = bias
for i, (h1_val, weight) in enumerate(zip(hidden1_post_activation, weights)):
    calculation_parts.append(f"({h1_val:.2f}×{weight:+.2f})")
    total += h1_val * weight

print(f"    = {' + '.join(calculation_parts[:4])} + ...")
print(f"    + bias({bias:+.2f})")
print(f"    = {total:.3f}")
print(f"    After ReLU: {max(0, total):.3f}")

print(f"\n  All Hidden Layer 2 activations:")
for i, val in enumerate(hidden2_post_activation):
    bar = "█" * int(min(val * 3, 40))
    print(f"    Neuron {i+1}: {val:6.3f} {bar}")

# LAYER 3: Hidden2 -> Output
print("\n" + "🔴 OUTPUT LAYER (2 neurons)".center(80))
print("="*80)
print("\nFinal layer produces driving commands (using tanh activation):\n")

hidden2_tensor = torch.FloatTensor(hidden2_post_activation)
output_pre_activation = layer3(hidden2_tensor).detach().numpy()
output_post_activation = np.tanh(output_pre_activation)  # Tanh

actions = ["Acceleration", "Steering"]
for i, action in enumerate(actions):
    print(f"  {action}:")
    weights = layer3.weight[i].detach().numpy()
    bias = layer3.bias[i].item()

    calculation_parts = []
    total = bias
    for h2_val, weight in zip(hidden2_post_activation, weights):
        calculation_parts.append(f"({h2_val:.2f}×{weight:+.2f})")
        total += h2_val * weight

    print(f"    = {' + '.join(calculation_parts[:3])} + ...")
    print(f"    + bias({bias:+.2f})")
    print(f"    = {total:.3f}")
    print(f"    After Tanh: {output_post_activation[i]:.3f}")

    # Interpret the output
    if i == 0:  # Acceleration
        if output_post_activation[i] > 0:
            action_str = f"🚗 Accelerate ({output_post_activation[i]:.1%} power)"
        else:
            action_str = f"🛑 Brake ({abs(output_post_activation[i]):.1%} power)"
    else:  # Steering
        if output_post_activation[i] > 0:
            action_str = f"➡️  Turn RIGHT ({output_post_activation[i]:.1%} intensity)"
        else:
            action_str = f"⬅️  Turn LEFT ({abs(output_post_activation[i]):.1%} intensity)"

    print(f"    → {action_str}\n")

print("="*80)
print("🎯 FINAL DECISION")
print("="*80)
print(f"\n  The car will:")
if output_post_activation[0] > 0:
    print(f"    • Press gas pedal {output_post_activation[0]:.1%}")
else:
    print(f"    • Press brake pedal {abs(output_post_activation[0]):.1%}")

if output_post_activation[1] > 0:
    print(f"    • Steer RIGHT at {output_post_activation[1]:.1%} intensity")
else:
    print(f"    • Steer LEFT at {abs(output_post_activation[1]):.1%} intensity")

print("\n" + "="*80)
print(f"💡 KEY INSIGHT: These {network.get_num_weights()} weights encode driving behavior!")
print("="*80)

# Now create a visual diagram
print("\n📊 Generating visual diagram...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Neural Network Visualization: Racing Car Brain', fontsize=16, fontweight='bold')

# ============ SUBPLOT 1: Network Architecture ============
ax1 = axes[0, 0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Network Architecture', fontsize=14, fontweight='bold')

# Draw layers
layer_x = [1, 3.5, 6, 8.5]
layer_sizes = [5, 8, 8, 2]
layer_names = ['Input\n(Sensors)', 'Hidden 1\n(ReLU)', 'Hidden 2\n(ReLU)', 'Output\n(Tanh)']
layer_colors = ['lightblue', 'lightgreen', 'lightgreen', 'salmon']

neurons_positions = []
for layer_idx, (x, size, name, color) in enumerate(zip(layer_x, layer_sizes, layer_names, layer_colors)):
    positions = []
    y_start = 5 - (size * 0.5)

    for i in range(size):
        y = y_start + i * (10 / (size + 1))
        circle = plt.Circle((x, y), 0.15, color=color, ec='black', linewidth=2, zorder=3)
        ax1.add_patch(circle)
        positions.append((x, y))

    neurons_positions.append(positions)
    ax1.text(x, 0.5, name, ha='center', fontsize=10, fontweight='bold')

# Draw connections (sample only for clarity)
for i in range(len(neurons_positions) - 1):
    for start_pos in neurons_positions[i][::2]:  # Every other neuron for clarity
        for end_pos in neurons_positions[i+1][::2]:
            ax1.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                    'gray', alpha=0.3, linewidth=0.5, zorder=1)

# ============ SUBPLOT 2: Activation Values ============
ax2 = axes[0, 1]
all_activations = {
    'Input': sensor_data,
    'Hidden 1': hidden1_post_activation,
    'Hidden 2': hidden2_post_activation,
    'Output': output_post_activation
}

y_pos = 0
colors_map = {'Input': 'skyblue', 'Hidden 1': 'lightgreen', 'Hidden 2': 'lightgreen', 'Output': 'salmon'}
for layer_name, values in all_activations.items():
    for i, val in enumerate(values):
        color = colors_map[layer_name]
        ax2.barh(y_pos, val, color=color, edgecolor='black', linewidth=1)
        ax2.text(val + 0.05, y_pos, f'{val:.2f}', va='center', fontsize=8)
        ax2.text(-0.15, y_pos, f'{layer_name} {i+1}', ha='right', va='center', fontsize=8)
        y_pos -= 1
    y_pos -= 0.5  # Gap between layers

ax2.set_xlim(-0.2, max(np.max(hidden1_post_activation), np.max(hidden2_post_activation)) + 0.5)
ax2.set_ylim(y_pos, 1)
ax2.set_xlabel('Activation Value', fontsize=10)
ax2.set_title('Neuron Activation Values', fontsize=14, fontweight='bold')
ax2.axvline(0, color='black', linewidth=0.5)
ax2.grid(axis='x', alpha=0.3)

# ============ SUBPLOT 3: Weight Matrix Heatmap ============
ax3 = axes[1, 0]
weights_layer1 = layer1.weight.detach().numpy()
im = ax3.imshow(weights_layer1, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
ax3.set_xlabel('Input Sensors', fontsize=10)
ax3.set_ylabel('Hidden Layer 1 Neurons', fontsize=10)
ax3.set_title('Layer 1 Weights (Input → Hidden 1)', fontsize=14, fontweight='bold')
ax3.set_xticks(range(5))
ax3.set_xticklabels(['-60°', '-30°', '0°', '+30°', '+60°'])
ax3.set_yticks(range(8))
ax3.set_yticklabels([f'H1-{i+1}' for i in range(8)])
plt.colorbar(im, ax=ax3, label='Weight Value')

# Add weight values as text
for i in range(8):
    for j in range(5):
        text = ax3.text(j, i, f'{weights_layer1[i, j]:.1f}',
                       ha="center", va="center", color="black", fontsize=7)

# ============ SUBPLOT 4: Sensor Data Visualization ============
ax4 = axes[1, 1]
angles = [-60, -30, 0, 30, 60]
distances = sensor_data

# Draw car
car_x, car_y = 0, 0
car = patches.Rectangle((car_x - 0.1, car_y - 0.15), 0.2, 0.3,
                         linewidth=2, edgecolor='blue', facecolor='lightblue', zorder=3)
ax4.add_patch(car)

# Draw sensors
for angle, distance in zip(angles, distances):
    angle_rad = np.radians(angle + 90)  # +90 because car faces up
    end_x = car_x + distance * np.cos(angle_rad)
    end_y = car_y + distance * np.sin(angle_rad)

    # Color based on distance (red = close, green = far)
    color = plt.cm.RdYlGn(distance)

    ax4.plot([car_x, end_x], [car_y, end_y], color=color, linewidth=3, zorder=2)
    ax4.scatter(end_x, end_y, s=100, color=color, edgecolor='black', linewidth=2, zorder=4)
    ax4.text(end_x + 0.1, end_y + 0.1, f'{distance:.2f}', fontsize=9, fontweight='bold')

ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-0.5, 1.5)
ax4.set_aspect('equal')
ax4.set_title('Sensor Readings (Car\'s View)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Distance (0=close, 1=far)', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)

# Add legend for output interpretation
output_text = f"OUTPUT:\nAccel: {output_post_activation[0]:+.2f}\nSteer: {output_post_activation[1]:+.2f}"
ax4.text(0.98, 0.02, output_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('neural_network_visualization.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved as 'neural_network_visualization.png'")
# plt.show()  # Commented out to avoid blocking

print("\n" + "="*80)
print("🎓 WHAT YOU JUST SAW:")
print("="*80)
print("""
1. INPUT LAYER: Raw sensor data from the car's 5 distance sensors

2. HIDDEN LAYER 1: Each neuron looks for specific patterns in the sensor data
   - Neurons multiply inputs by weights (importance)
   - Add a bias (baseline value)
   - Apply ReLU (make negative values zero)

3. HIDDEN LAYER 2: Combines patterns from Layer 1 into higher-level features
   - Same process: weights → sum → bias → ReLU

4. OUTPUT LAYER: Produces final driving commands
   - Uses Tanh to squash values to range [-1, 1]
   - Acceleration: negative = brake, positive = gas
   - Steering: negative = left, positive = right

The network has 138 numbers (weights + biases) that were learned through
evolution. These numbers encode the car's driving strategy!
""")
print("="*80)
