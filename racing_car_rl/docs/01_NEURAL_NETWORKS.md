# Neural Networks - The Car's Brain

## What Is a Neural Network?

A neural network is a mathematical function that learns to map inputs to outputs. Think of it as a "brain" that makes decisions.

### Simple Analogy

**Your Brain:**
- Eyes see wall → Brain processes → Hands turn steering wheel

**Car's Neural Network:**
- Sensors detect wall → Network processes → Output steering action

## Our Network Architecture

```
Input Layer (5 neurons)          Hidden Layer 1 (8 neurons)
┌─────────────┐                 ┌─────┐
│ Sensor 1    │───┐         ┌──▶│ H1  │
│ (-60°)      │   │         │   └─────┘
└─────────────┘   │         │   ┌─────┐
                  ├────────▶├──▶│ H2  │
┌─────────────┐   │         │   └─────┘
│ Sensor 2    │───┤         │   ┌─────┐
│ (-30°)      │   │         ├──▶│ H3  │
└─────────────┘   │         │   └─────┘
                  │         │     ...
┌─────────────┐   │         │   ┌─────┐
│ Sensor 3    │───┤         └──▶│ H8  │
│ (0°)        │   │             └─────┘
└─────────────┘   │                │
                  │                │
┌─────────────┐   │                ▼
│ Sensor 4    │───┤         Hidden Layer 2 (8 neurons)
│ (+30°)      │   │             ┌─────┐
└─────────────┘   │         ┌──▶│ H1  │
                  │         │   └─────┘
┌─────────────┐   │         │     ...
│ Sensor 5    │───┘         │   ┌─────┐
│ (+60°)      │             └──▶│ H8  │
└─────────────┘                 └─────┘
                                   │
                                   ▼
                          Output Layer (2 neurons)
                            ┌───────────────┐
                            │ Acceleration  │ (-1 to +1)
                            │ (gas/brake)   │
                            └───────────────┘
                            ┌───────────────┐
                            │ Steering      │ (-1 to +1)
                            │ (left/right)  │
                            └───────────────┘
```

## How It Works - Step by Step

### Step 1: Sensors Collect Data

```python
sensor_data = [0.8, 0.6, 0.9, 0.7, 0.5]
# Each value is distance to wall (0 = very close, 1 = far away)
```

**Example Scenario:**
- Sensor 1 (-60°): 0.8 → Wall is far to the left
- Sensor 2 (-30°): 0.6 → Wall is medium distance left-front
- Sensor 3 (0°): 0.9 → Wall is far ahead
- Sensor 4 (+30°): 0.7 → Wall is medium-far right-front
- Sensor 5 (+60°): 0.5 → Wall is CLOSE to the right!

### Step 2: First Hidden Layer

Each hidden neuron computes:

```python
# Simplified explanation
hidden1_neuron1 = ReLU(
    sensor1 * weight1 +
    sensor2 * weight2 +
    sensor3 * weight3 +
    sensor4 * weight4 +
    sensor5 * weight5 +
    bias
)
```

**What's happening:**
- Each sensor value is multiplied by a **weight** (importance)
- All weighted values are summed together
- A **bias** is added (like a baseline)
- **ReLU** activation makes it non-linear (if negative, becomes 0)

### Step 3: Second Hidden Layer

Same process, but uses outputs from first hidden layer:

```python
hidden2_neuron1 = ReLU(
    hidden1_1 * weight +
    hidden1_2 * weight +
    ... (all 8 hidden1 neurons)
)
```

### Step 4: Output Layer

Final layer produces driving commands:

```python
acceleration = tanh(hidden2_neurons * weights)  # Value: -1 to +1
steering = tanh(hidden2_neurons * weights)      # Value: -1 to +1
```

**Tanh activation:**
- Squashes output to range [-1, 1]
- Negative values mean brake/turn left
- Positive values mean accelerate/turn right

### Step 5: Car Takes Action

```python
if acceleration > 0:
    car.accelerate(acceleration)  # Gas pedal
else:
    car.brake(abs(acceleration))  # Brake pedal

if steering > 0:
    car.turn_right(steering)
else:
    car.turn_left(abs(steering))
```

## Weights - The Learning Part

### What Are Weights?

Weights are numbers that determine how important each connection is.

**Example:**
```python
# If sensor 3 (front) has a high weight to "brake" neuron:
weight_sensor3_to_brake = 5.2

# And sensor 3 detects wall close (0.1):
brake_signal = 0.1 * 5.2 = 0.52  # Strong brake signal!
```

**Good weights produce good driving:**
- Wall ahead → Brake and turn
- Clear path → Accelerate
- Wall on right → Turn left

**Bad weights produce crashes:**
- Wall ahead → Accelerate (CRASH!)
- Clear path → Brake (slow/stopped)
- Random steering (zigzag everywhere)

### How Many Weights Does Our Network Have?

```
Layer 1: 5 inputs × 8 neurons = 40 weights + 8 biases = 48
Layer 2: 8 inputs × 8 neurons = 64 weights + 8 biases = 72
Layer 3: 8 inputs × 2 outputs = 16 weights + 2 biases = 18
───────────────────────────────────────────────────────────
Total: 138 learnable parameters
```

Each car has 138 numbers that define its behavior!

## Why Hidden Layers?

### Without Hidden Layers (Linear):
```
Sensors → Output
```
Can only learn simple patterns:
- "If wall ahead, turn"

### With Hidden Layers (Non-linear):
```
Sensors → Hidden → Hidden → Output
```
Can learn complex behaviors:
- "If wall ahead AND moving fast, brake hard AND turn"
- "If wall on right but far ahead is clear, gentle right turn"
- "If surrounded by walls, sharp emergency turn"

## Activation Functions

### ReLU (Rectified Linear Unit)
```python
def relu(x):
    return max(0, x)
```

**Example:**
- Input: -2 → Output: 0
- Input: 3 → Output: 3

**Why use it:**
- Prevents negative values in hidden layers
- Computationally efficient
- Helps network learn complex patterns

### Tanh (Hyperbolic Tangent)
```python
def tanh(x):
    return (e^x - e^-x) / (e^x + e^-x)
```

**Example:**
- Input: -10 → Output: -1
- Input: 0 → Output: 0
- Input: 10 → Output: 1

**Why use it:**
- Perfect for actions that go both ways (steer left/right, brake/gas)
- Smooth gradients
- Centered around 0

## A Concrete Example

Let's trace one car's decision:

**Scenario: Approaching a left turn**

1. **Sensor readings:**
   ```python
   sensors = [0.3, 0.4, 0.6, 0.8, 0.9]
   # Left sensors (0.3, 0.4) = walls close on left (turn coming up)
   # Front (0.6) = some space ahead
   # Right sensors (0.8, 0.9) = clear on right
   ```

2. **Hidden layer 1 processes:**
   ```python
   # Neuron 1 might specialize in detecting "left wall approaching"
   hidden1[0] = ReLU(0.3*2.1 + 0.4*1.8 + 0.6*0.5 + ...) = 2.45

   # Neuron 2 might detect "right side clear"
   hidden1[1] = ReLU(...0.8*2.3 + 0.9*2.1...) = 4.12
   ```

3. **Hidden layer 2 combines:**
   ```python
   # Neuron specializing in "turn decision"
   hidden2[3] = ReLU(2.45*1.2 + 4.12*1.5 + ...) = 5.67
   ```

4. **Output layer decides:**
   ```python
   acceleration = tanh(...) = 0.3   # Slight gas
   steering = tanh(5.67*0.8) = 0.6  # Turn right (away from left wall)
   ```

5. **Car executes:**
   - Maintains moderate speed (0.3 acceleration)
   - Steers right (0.6 steering) to follow the turn
   - Successfully navigates the corner!

## What the Network Is Learning

Over generations, the network learns patterns like:

```
Pattern 1: "Wall Detection"
IF front_sensor < 0.4 THEN brake_hard

Pattern 2: "Turn Navigation"
IF left_sensors < right_sensors THEN steer_right

Pattern 3: "Speed Control"
IF all_sensors > 0.7 THEN accelerate

Pattern 4: "Emergency Avoidance"
IF any_sensor < 0.2 THEN brake_and_turn_away
```

But it doesn't learn these as explicit rules! The weights encode these behaviors implicitly through 138 numbers.

## Code Implementation

Our network in `neural_network.py`:

```python
class CarNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 8),    # Input → Hidden 1
            nn.ReLU(),          # Activation
            nn.Linear(8, 8),    # Hidden 1 → Hidden 2
            nn.ReLU(),          # Activation
            nn.Linear(8, 2),    # Hidden 2 → Output
            nn.Tanh()           # Output activation
        )
```

**This defines:**
- Structure (5→8→8→2)
- Activations (ReLU, Tanh)
- Initializes 138 random weights

## Key Takeaways

1. **Neural networks transform inputs to outputs** through layers of weighted connections

2. **Weights are the "knowledge"** - they encode what the network has learned

3. **Activations add non-linearity** - allowing complex decision-making

4. **Our network is small (138 parameters)** - but sufficient for this task

5. **No explicit programming** - the network figures out driving strategies through evolution

## Next: How Do Weights Get Better?

In traditional deep learning, we use **backpropagation** with gradient descent.

In our project, we use **genetic algorithms** - evolution!

Continue to **02_GENETIC_ALGORITHMS.md** to learn how! 🧬
