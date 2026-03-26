# Quick Reference Guide

## Glossary of Terms

### Core Concepts

**Agent** - The car that learns to drive (also called "actor" or "bot")

**Environment** - The race track where learning happens

**State** - Current sensor readings [s1, s2, s3, s4, s5]

**Action** - What the car does [acceleration, steering]

**Fitness** - Score measuring performance (distance traveled)

**Population** - Group of cars (currently 5)

**Generation** - One cycle of evaluation → evolution

**Elite** - Best performer(s) copied to next generation unchanged

**Mutation** - Random changes to network weights

**Crossover** - Combining weights from two parent networks

**Neural Network** - Mathematical function that maps sensors → actions

**Weights** - 138 numbers that define a network's behavior

**Epoch** - Not used (that's for gradient descent training)

**Gradient** - Not used (we're using evolution, not backpropagation)

### Mathematical Notation

```
s = sensors (5-dimensional vector)
a = actions (2-dimensional vector)
θ = network weights (138-dimensional vector)
f(θ) = fitness of network with weights θ
π(s;θ) = policy: sensors → actions
```

## Key Equations

### Neural Network Forward Pass

```
h₁ = ReLU(W₁·s + b₁)        Layer 1: 5 → 8
h₂ = ReLU(W₂·h₁ + b₂)       Layer 2: 8 → 8
a = tanh(W₃·h₂ + b₃)        Output: 8 → 2
```

### Fitness Function

```
f = d + (10 if alive else 0)

where:
  d = distance traveled
  alive = survived until end
```

### Mutation

```
θ' = θ + ε·N(0, σ²)

where:
  ε = mutation rate (0.1)
  σ = mutation scale (0.3)
  N = normal distribution
```

### Crossover

```
θ_child = [θ_parent1[0:k], θ_parent2[k:n]]

where:
  k = random crossover point
  n = total weights (138)
```

## File Reference

### track.py

```python
Track()
  .is_on_track(x, y) → bool     # Collision check
  .render(screen)                # Draw track
  .start_pos                     # (600, 650)
  .start_angle                   # -90 degrees
```

### car.py

```python
Car(x, y, angle, track)
  .update(action)                # Move car
  .get_sensor_data() → array[5]  # Raycast sensors
  .render(screen, show_sensors)  # Draw car
  .alive                         # bool
  .fitness                       # float
  .distance_traveled             # float
  .velocity                      # 0-15
```

### neural_network.py

```python
CarNeuralNetwork()
  .predict(sensors) → action     # Get action from sensors
  .get_weights() → array[138]    # Export weights
  .set_weights(array)            # Import weights
  .mutate(rate, scale)           # Random mutations
  .crossover(parent1, parent2)   # Combine networks
  .copy()                        # Deep copy
```

### genetic_algorithm.py

```python
GeneticAlgorithm(pop_size, elite_size, mutation_rate)
  .evolve(fitness_scores)        # Create next generation
  .get_population()              # Get all networks
  .get_statistics()              # Get training stats
  .save_best(filename)           # Save to disk
  .load_best(filename)           # Load from disk
```

### main.py

```python
# Key parameters (top of file)
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
SIMULATION_SPEED = 4           # Physics updates per frame
MAX_GENERATION_TIME = 5        # Seconds per generation
POPULATION_SIZE = 5            # Cars per generation

# Main loop
for each generation:
    run_generation()           # Simulate 5 cars
    ga.evolve(fitness_scores)  # Create next gen
```

## Common Modifications

### Change Population Size

```python
# main.py
POPULATION_SIZE = 10  # From 5 to 10

# Also update elite size proportionally
elite_size = max(1, POPULATION_SIZE // 5)
```

### Adjust Speed

```python
# main.py
SIMULATION_SPEED = 8        # 8x speed (from 4x)
MAX_GENERATION_TIME = 3     # 3 seconds (from 5)

# car.py
self.max_velocity = 20      # Faster cars (from 15)
```

### Modify Fitness Function

```python
# car.py, in _update_fitness()
def _update_fitness(self):
    self.fitness = self.distance_traveled

    # Add survival bonus
    if self.alive:
        self.fitness += 10

    # Add speed reward (NEW)
    self.fitness += self.velocity * 0.5

    # Add smoothness reward (NEW)
    self.fitness -= abs(self.last_steering - self.current_steering)
```

### Change Network Architecture

```python
# neural_network.py
def __init__(self):
    self.network = nn.Sequential(
        nn.Linear(5, 16),   # Bigger hidden layer (from 8)
        nn.ReLU(),
        nn.Linear(16, 16),  # Bigger hidden layer (from 8)
        nn.ReLU(),
        nn.Linear(16, 2),
        nn.Tanh()
    )
```

### Add More Sensors

```python
# car.py
self.sensor_angles = [-90, -60, -30, 0, 30, 60, 90]  # 7 sensors (from 5)

# neural_network.py - update input size
nn.Linear(7, 8)  # Input size 7 (from 5)
```

### Adjust Mutation

```python
# main.py
ga = GeneticAlgorithm(
    population_size=5,
    elite_size=1,
    mutation_rate=0.2  # More exploration (from 0.1)
)

# neural_network.py - in mutate()
mutation_scale = 0.5  # Larger changes (from 0.3)
```

## Troubleshooting Checklist

### Cars Not Learning

- [ ] Check fitness is increasing over generations
- [ ] Verify elite is preserved (best shouldn't decrease)
- [ ] Ensure sensors are working (visualize them)
- [ ] Confirm mutation rate isn't too high
- [ ] Check network isn't too small

### Simulation Too Slow

- [ ] Reduce POPULATION_SIZE
- [ ] Increase SIMULATION_SPEED
- [ ] Decrease MAX_GENERATION_TIME
- [ ] Disable sensor rendering

### Simulation Too Fast

- [ ] Reduce SIMULATION_SPEED
- [ ] Increase MAX_GENERATION_TIME
- [ ] Lower FPS

### Cars Stuck/Not Improving

- [ ] Increase mutation rate temporarily
- [ ] Add random individual to population
- [ ] Check if stuck in local optimum
- [ ] Try different random seed

### Crashes/Errors

- [ ] Check all coordinates are integers for pygame
- [ ] Verify tensor shapes match
- [ ] Ensure weights array length is correct
- [ ] Check division by zero in raycasting

## Performance Benchmarks

### Expected Learning Curve

```
Gen 0-10:   Fitness 0-300     (Learning basics)
Gen 10-30:  Fitness 300-700   (Competent driving)
Gen 30-60:  Fitness 700-1200  (Skilled driving)
Gen 60+:    Fitness 1200+     (Expert level)
```

### Training Time

```
With current settings:
- Time per generation: ~5 seconds
- 50 generations: ~4 minutes
- 100 generations: ~8 minutes
```

### Computational Cost

```
Per generation:
- Neural network forward passes: 5 cars × 1200 updates = 6,000
- Raycasts: 5 sensors × 5 cars × 1200 updates = 30,000
- Physics updates: 5 cars × 1200 updates = 6,000
```

## Keyboard Shortcuts

### During Training (main.py)

```
ESC   - Quit training
SPACE - Skip to next generation
```

### During Demo (demo.py)

```
ESC   - Quit demo
R     - Reset car to start
```

## Command Line Usage

```bash
# Training
python main.py

# Demo with default network
python demo.py

# Demo with specific network
python demo.py best_network_gen_50.pth

# Install dependencies
uv sync --no-install-project
pip install -r requirements.txt
```

## File Outputs

### Generated Files

```
best_network_final.pth         # Final best network
best_network_gen_X.pth         # Network at generation X (every 10)
fitness_history.png            # Training graph
uv.lock                        # Dependency lock file
```

### Loading Saved Networks

```python
from genetic_algorithm import GeneticAlgorithm

ga = GeneticAlgorithm()
network = ga.load_best('best_network_gen_50.pth')

# Use it
action = network.predict(sensors)
```

## Parameter Tuning Guide

### Quick Reference Table

| Parameter | Too Low | Sweet Spot | Too High |
|-----------|---------|------------|----------|
| Population | No diversity (1-2) | 5-10 | Slow training (50+) |
| Elite Size | Lose progress (0) | 1-2 | No evolution (5) |
| Mutation Rate | Too slow (0.01) | 0.1-0.2 | Chaos (0.5) |
| Mutation Scale | No change (0.01) | 0.2-0.4 | Destruction (1.0) |
| Hidden Neurons | Can't learn (2) | 8-16 | Slow evolve (64) |
| Max Velocity | Too slow (3) | 8-15 | Hard to control (30) |
| Sensor Length | Blind (50px) | 150-250px | Expensive (500px) |
| Generation Time | Too short (1s) | 5-10s | Waste time (30s) |

## Experiment Ideas

### Easy Modifications

1. Change car colors
2. Adjust speed multiplier
3. Modify track shape
4. Add more/fewer cars
5. Change generation time

### Medium Difficulty

1. Add velocity to neural network input
2. Implement different track layouts
3. Add checkpoints for fitness
4. Save/load populations
5. Add multiple tracks for training

### Advanced Projects

1. Implement memory (LSTM)
2. Add competing cars
3. Co-evolve tracks and cars
4. Multi-objective optimization
5. Transfer learning across tracks
6. Compare to DQN/PPO implementations

## Useful Code Snippets

### Print Detailed Stats

```python
# In run_generation(), after simulation
for i, car in enumerate(cars):
    print(f"Car {i}: Fitness={car.fitness:.1f}, "
          f"Distance={car.distance_traveled:.1f}, "
          f"Alive={car.alive}")
```

### Save Every Generation

```python
# In main loop
ga.save_best(f'networks/gen_{generation}.pth')
```

### Plot Real-time Fitness

```python
import matplotlib.pyplot as plt

# After each generation
plt.clf()
plt.plot(ga.best_fitness_history)
plt.pause(0.01)
```

### Headless Training (No GUI)

```python
# Remove all pygame rendering
# Use track logic but skip display
# 10x faster training!

for car in cars:
    while car.alive:
        sensors = car.get_sensor_data()
        action = network.predict(sensors)
        car.update(action)
```

### Test on Multiple Tracks

```python
tracks = [SimpleTrack(), ComplexTrack(), Figure8()]

for track in tracks:
    fitness = evaluate_network(network, track)
    print(f"Track {track.name}: {fitness}")
```

## Resources

### Documentation
- `docs/00_OVERVIEW.md` - Project introduction
- `docs/01_NEURAL_NETWORKS.md` - Brain explanation
- `docs/02_GENETIC_ALGORITHMS.md` - Evolution theory
- `docs/03_SENSOR_SYSTEM.md` - Perception system
- `docs/04_LEARNING_PROCESS.md` - How learning works
- `docs/05_ADVANCED_CONCEPTS.md` - Deep dive

### External Learning
- PyTorch tutorials: pytorch.org/tutorials
- Genetic algorithms: wikipedia.org/wiki/Genetic_algorithm
- Reinforcement learning: sutton-barto.com

---

**Quick Start:** `python main.py` → Watch cars evolve! 🚗💨
