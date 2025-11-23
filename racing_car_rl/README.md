# Racing Car Reinforcement Learning with Genetic Algorithm

A deep learning project where autonomous cars learn to navigate a race track using neural networks and genetic algorithms. Watch as cars evolve over generations, learning to complete the track without crashing!

## 📚 Learn the Theory

**New to machine learning or genetic algorithms?** Start here:

1. **[Overview](docs/00_OVERVIEW.md)** - Big picture and key concepts
2. **[Neural Networks](docs/01_NEURAL_NETWORKS.md)** - How the car's brain works
3. **[Genetic Algorithms](docs/02_GENETIC_ALGORITHMS.md)** - How evolution happens
4. **[Sensor System](docs/03_SENSOR_SYSTEM.md)** - How cars "see" the track
5. **[Learning Process](docs/04_LEARNING_PROCESS.md)** - How it all comes together
6. **[Advanced Concepts](docs/05_ADVANCED_CONCEPTS.md)** - Deep dive into code
7. **[Quick Reference](docs/QUICK_REFERENCE.md)** - Cheat sheet

## Overview

This project demonstrates **neuroevolution** - training neural networks using genetic algorithms instead of traditional backpropagation. Cars equipped with distance sensors use small neural networks to make driving decisions, and the best-performing cars pass their "genes" (network weights) to the next generation.

### Key Features

- **Real-time visualization** of 5 cars learning simultaneously
- **Physics simulation** with acceleration, steering, and collision detection
- **5 raycast sensors** for detecting track boundaries
- **Genetic algorithm** with elitism, crossover, and mutation
- **PyTorch neural networks** for decision making
- **Fitness tracking** and progress visualization

## How It Works

### 1. Neural Network Architecture

Each car has a simple feedforward neural network:
- **Input**: 5 distance sensors (normalized 0-1)
- **Hidden Layers**: 2 layers with 8 neurons each, ReLU activation
- **Output**: 2 actions (acceleration and steering), values in [-1, 1]

```
Sensors (5) → Hidden (8) → Hidden (8) → Actions (2)
```

### 2. Genetic Algorithm Process

Each generation follows these steps:

1. **Evaluation**: All 50 cars run simultaneously until they crash or time runs out
2. **Fitness Calculation**: Based on distance traveled + survival bonus
3. **Selection**: Top 5 cars (elite) move to next generation unchanged
4. **Crossover**: Pairs of successful cars create offspring by mixing their network weights
5. **Mutation**: Random changes to weights (10% probability) add diversity
6. **Next Generation**: Process repeats with improved population

### 3. Car Sensors

Each car has 5 distance sensors at angles: -60°, -30°, 0°, 30°, 60°

- Sensors cast rays to detect track boundaries
- Return normalized distance (0 = collision, 1 = max range)
- Neural network uses these to decide acceleration and steering

### 4. Fitness Function

```python
fitness = distance_traveled + (10 if alive else 0)
```

Cars are rewarded for:
- Traveling further around the track
- Staying alive longer

## Installation

1. **Clone or navigate to the project directory**:
```bash
cd racing_car_rl
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- pygame >= 2.5.0
- numpy >= 1.24.0
- torch >= 2.0.0
- matplotlib >= 3.7.0

## Usage

### Training

Run the main training script:

```bash
python main.py
```

**Controls during training**:
- `ESC`: Quit training
- `SPACE`: Skip to next generation

The simulation will:
- Display 5 cars learning in real-time at 4x speed
- Show generation number, alive count, and progress bar
- Print statistics after each generation
- Save best network every 10 generations
- Save final network as `best_network_final.pth`
- Generate a fitness plot when finished

### Demo Mode

Watch a trained network in action:

```bash
python demo.py
```

Or specify a specific saved network:

```bash
python demo.py best_network_gen_50.pth
```

**Controls during demo**:
- `ESC`: Quit
- `R`: Reset car to start position

## Project Structure

```
racing_car_rl/
├── main.py                 # Training loop with visualization
├── demo.py                 # Demo script for trained networks
├── track.py                # Race track environment
├── car.py                  # Car physics and sensors
├── neural_network.py       # Neural network implementation
├── genetic_algorithm.py    # Genetic algorithm for evolution
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
├── README.md               # This file
└── docs/                   # Educational materials (START HERE!)
    ├── 00_OVERVIEW.md
    ├── 01_NEURAL_NETWORKS.md
    ├── 02_GENETIC_ALGORITHMS.md
    ├── 03_SENSOR_SYSTEM.md
    ├── 04_LEARNING_PROCESS.md
    ├── 05_ADVANCED_CONCEPTS.md
    └── QUICK_REFERENCE.md
```

## Understanding the Code

### track.py
Defines an oval race track with:
- Outer and inner boundaries
- Collision detection
- Rendering functions

### car.py
Implements car behavior:
- Physics (velocity, acceleration, friction)
- 5 raycast sensors for boundary detection
- Fitness tracking
- Rendering with sensor visualization

### neural_network.py
PyTorch neural network with:
- Forward pass for decision making
- Weight getting/setting for genetic algorithm
- Mutation and crossover operations
- Network copying and initialization

### genetic_algorithm.py
Evolution logic:
- Population management
- Tournament selection
- Crossover and mutation
- Fitness tracking and statistics

### main.py
Main training loop:
- Pygame visualization
- Generation simulation
- UI rendering
- Progress tracking

## Tuning Parameters

You can modify these parameters in `main.py`:

```python
POPULATION_SIZE = 50          # Number of cars per generation
MAX_GENERATION_TIME = 15      # Seconds per generation
FPS = 60                      # Frames per second
```

In `genetic_algorithm.py`:

```python
elite_size = 5                # Top performers kept unchanged
mutation_rate = 0.1           # Probability of weight mutation
```

In `neural_network.py`:

```python
hidden_size = 8               # Neurons in hidden layers
mutation_scale = 0.3          # Size of random mutations
```

## Expected Results

### Early Generations (0-10)
- Cars drive randomly and crash quickly
- Most barely move from start position
- Fitness scores: 50-200

### Mid Training (10-30)
- Cars learn basic steering
- Some navigate first few turns
- Fitness scores: 200-800

### Late Training (30+)
- Best cars complete full laps
- Smooth, efficient driving
- Fitness scores: 800-2000+

## Learning Resources

This project demonstrates:
- **Genetic Algorithms**: Evolution-based optimization
- **Neuroevolution**: Evolving neural networks without backpropagation
- **Reinforcement Learning**: Learning through trial and error
- **PyTorch**: Deep learning framework
- **Pygame**: Game development and visualization

## Extending the Project

Ideas for improvements:
1. **More complex tracks**: Add curves, obstacles, or multiple routes
2. **Better sensors**: Add velocity sensors, track position detection
3. **Advanced GA**: Try NEAT (evolving network topology too)
4. **Hybrid training**: Combine genetic algorithm with gradient descent
5. **Multiple tracks**: Train on diverse tracks for generalization
6. **Replay system**: Save and replay best runs
7. **Competitive mode**: Race multiple trained networks

## Troubleshooting

**Cars don't improve**:
- Increase population size
- Adjust mutation rate (try 0.05-0.2)
- Run more generations
- Increase generation time

**Simulation too slow**:
- Reduce population size
- Lower FPS
- Disable sensor visualization

**Cars learn then get worse**:
- Reduce mutation rate
- Increase elite size
- Add weight clipping in mutation

## License

This project is open source and available for educational purposes.

## Acknowledgments

Inspired by:
- NEAT (NeuroEvolution of Augmenting Topologies)
- Traditional genetic algorithms
- Self-driving car simulations
- Reinforcement learning research

---

**Have fun watching your cars learn to race!** 🏎️
