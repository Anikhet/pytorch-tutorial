# Flappy Bird AI - Genetic Algorithm

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Optional-yellow)

Watch AI birds learn to play Flappy Bird using **genetic algorithms** and **neural networks**! This project demonstrates neuroevolution, where neural network weights evolve over generations without traditional backpropagation.

## Learning Objectives

By completing this tutorial, you will learn:

- **Neuroevolution**: Evolve neural network weights without gradients
- **Tournament Selection**: Select parents based on competitive fitness
- **Crossover Operators**: Combine parent networks to create offspring
- **Mutation Strategies**: Add random perturbations for exploration
- **Fitness Landscapes**: Understand how fitness guides evolution
- **Game AI**: Apply neural networks to real-time decision making

## Overview

50 birds start with random neural networks. The birds that survive longer and score more points pass their "brain" genes to the next generation. Over time, the population evolves to master the game!

### Key Features

- **Neural Networks**: Each bird has a PyTorch neural network brain (4→8→1)
- **Genetic Algorithm**: Tournament selection, crossover, and mutation
- **Real-time Visualization**: Watch all 50 birds learning simultaneously
- **Evolution Tracking**: Graphs showing fitness and score improvement
- **Demo Mode**: Showcase your best trained bird

## How It Works

### 1. Neural Network (The Bird's Brain)

Each bird uses a simple feedforward neural network to decide when to jump:

**Input Layer (4 neurons)**:
- Bird's Y position (normalized 0-1)
- Horizontal distance to next pipe (normalized)
- Y position of gap center (normalized)
- Bird's velocity (normalized)

**Hidden Layer (8 neurons)**:
- ReLU activation

**Output Layer (1 neuron)**:
- Sigmoid activation → jump probability
- If output > 0.5, the bird jumps!

### 2. Genetic Algorithm (Evolution)

**Population**: 50 birds per generation

**Fitness Function**:
```
fitness = frames_alive + score × 100
```

**Evolution Steps**:
1. **Evaluation**: All birds play until they die
2. **Selection**: Best birds chosen using tournament selection
3. **Elitism**: Top 5 birds kept unchanged
4. **Crossover**: Parent brains combine to create children
5. **Mutation**: Random weight changes (10% probability)

### 3. Game Mechanics

- **Pipes**: Green obstacles with 150-pixel gaps
- **Physics**: Gravity pulls bird down, jump gives upward velocity
- **Collision**: Bird dies if hitting pipe, ground, or ceiling
- **Scoring**: +1 point per pipe successfully passed

## Installation

### Requirements
- Python 3.8+
- PyTorch
- Pygame
- NumPy
- Matplotlib

### Install Dependencies

```bash
cd flappy_bird_ai
pip install -r requirements.txt
```

## Usage

### Training Mode

Train birds from scratch:

```bash
python main.py
```

**What you'll see**:
- Yellow birds flying and learning
- Generation counter
- Alive count (birds still flying)
- Frame counter

**Controls**:
- `SPACE`: Skip to next generation
- `ESC`: Quit training

**Training will run for up to 100 generations or until a bird scores 50+ points.**

### Demo Mode

Watch your best trained bird:

```bash
# Use the final best bird
python demo.py

# Or specify a saved model
python demo.py best_bird_gen_50.pth
```

**Controls**:
- `SPACE`: Restart the demo
- `P`: Pause/Unpause
- `ESC`: Quit

## Files

```
flappy_bird_ai/
├── main.py                 # Training loop
├── demo.py                 # Demo trained birds
├── bird.py                 # Bird class with physics and brain
├── pipe.py                 # Pipe obstacles and manager
├── neural_network.py       # PyTorch neural network
├── genetic_algorithm.py    # GA implementation
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Configuration

Edit `main.py` to customize training:

```python
# Population settings
POPULATION_SIZE = 50    # Number of birds per generation
ELITE_SIZE = 5          # Best birds kept unchanged

# Evolution settings
MUTATION_RATE = 0.1     # Probability of mutating each weight
MAX_GENERATIONS = 100   # Maximum generations to train

# Display settings
FPS = 60                # Frames per second
```

## Understanding the Output

### During Training

```
Generation 0
Best Score: 3
Best Fitness: 487.00
Average Fitness: 234.56
Alive at end: 0
```

- **Best Score**: Highest number of pipes passed
- **Best Fitness**: frames_alive + score × 100
- **Average Fitness**: Mean fitness across all 50 birds
- **Alive at end**: Birds still alive when generation ended (usually 0)

### After Training

A progress plot is saved as `flappy_bird_progress.png` showing:
- **Best Fitness**: Evolution of the best bird's fitness
- **Average Fitness**: Population average over time
- **Best Score**: Highest score achieved each generation

## Expected Results

### Early Generations (0-5)
- Birds die almost immediately
- Most can't avoid first pipe
- Fitness: 50-200

### Middle Generations (5-20)
- Some birds pass 1-3 pipes
- Learning basic jump timing
- Fitness: 200-500

### Late Generations (20-50+)
- Birds consistently pass 5+ pipes
- Some may reach 20+ pipes
- Fitness: 500-2000+

### Mastery (50-100+)
- Birds can potentially play indefinitely
- Scores of 50+ achievable
- Fitness: 5000+

## Technical Details

### Neural Network Architecture

```python
BirdNeuralNetwork(
    input_size=4,
    hidden_size=8,
    output_size=1
)

Total parameters: 49
- Layer 1: 4×8 + 8 = 40 weights
- Layer 2: 8×1 + 1 = 9 weights
```

### Genetic Algorithm Parameters

- **Tournament Size**: 5 birds compete for selection
- **Crossover Method**: Uniform crossover (50% from each parent)
- **Mutation Scale**: σ = 0.3 (Gaussian noise)
- **Elite Preservation**: Top 5 birds (10% of population)

### Physics Constants

```python
# Bird physics
gravity = 0.5
jump_strength = -10
max_velocity = 10

# Pipe settings
gap_size = 150
pipe_speed = 3
spawn_distance = 250
```

## Why Genetic Algorithms?

Unlike traditional supervised learning, we don't have labeled data (no "correct" actions). Genetic algorithms are perfect for this because:

1. **No Training Data Needed**: Birds learn through trial and error
2. **Exploration**: Mutation ensures diverse strategies
3. **Simplicity**: No gradients, backprop, or optimizers needed
4. **Interpretable**: Easy to understand what's being optimized

## Extending the Project

Ideas to try:

1. **Harder Game**: Narrower gaps, faster pipes
2. **Smarter Birds**: Add more hidden layers or neurons
3. **Better Selection**: Try roulette wheel or rank-based selection
4. **Visualization**: Show bird's neural network activations
5. **Adaptive Difficulty**: Increase speed as birds improve
6. **Multiple Outputs**: Control jump force, not just jump/no-jump

## Saved Models

The training automatically saves:
- `best_bird_gen_10.pth`: Best bird every 10 generations
- `best_bird_gen_20.pth`
- `best_bird_final.pth`: Final best bird when training ends

Load them using:
```python
from genetic_algorithm import GeneticAlgorithm
ga = GeneticAlgorithm()
bird = ga.load_best('best_bird_final.pth')
```

## Troubleshooting

**Birds all die immediately**
- This is normal for early generations!
- They have random brains and need time to evolve
- Give it 10-20 generations

**Training is too slow**
- Reduce FPS in main.py
- Reduce POPULATION_SIZE
- Run in headless mode (disable rendering)

**Birds not improving after many generations**
- Try increasing MUTATION_RATE (more exploration)
- Try increasing POPULATION_SIZE (more diversity)
- Check if birds are getting stuck in local optimum

**Game crashes**
- Make sure all dependencies are installed
- Check Python version (3.8+)
- Verify pygame is working: `python -c "import pygame; pygame.init()"`

## Comparison to Other Learning Methods

| Method | Training Time | Data Needed | Interpretability |
|--------|--------------|-------------|------------------|
| **Genetic Algorithm** | Moderate | None | High |
| Deep Q-Learning | Fast | None | Low |
| Supervised Learning | Very Fast | Lots | Medium |
| Policy Gradient | Slow | None | Medium |

## Learn More

### Genetic Algorithms
- [Introduction to Genetic Algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [Neuroevolution](https://en.wikipedia.org/wiki/Neuroevolution)

### Similar Projects
- NEAT (NeuroEvolution of Augmenting Topologies)
- MarI/O (Mario AI using NEAT)
- SethBling's MarI/O video

## License

This project is for educational purposes. Feel free to modify and experiment!

## Author

Created as a demonstration of genetic algorithms and neuroevolution in game AI.

Happy evolving! Watch those birds learn! 🐦
