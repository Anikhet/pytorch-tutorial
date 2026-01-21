# Guitar Learning AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Optional-yellow)

A neural network that learns to play guitar solos using genetic algorithms! Watch as bots evolve over generations to master a rock-style guitar solo, learning the right strings, frets, and timing through natural selection.

## Learning Objectives

By completing this tutorial, you will learn:

- **Neuroevolution**: Train neural networks without backpropagation using genetic algorithms
- **Sequence Learning**: Model temporal dependencies in music note sequences
- **Fitness Function Design**: Create multi-objective fitness (timing, pitch, string selection)
- **Population Dynamics**: Implement selection pressure and genetic diversity
- **Elitism**: Preserve top performers across generations
- **Audio Synthesis**: Generate guitar sounds from neural network outputs

## Overview

This project demonstrates **neuroevolution** - training neural networks using genetic algorithms instead of traditional backpropagation. A population of "guitar players" compete to play a guitar solo correctly, with the best performers surviving and reproducing to create the next generation.

### Key Features

- **Real-time Visualization**: Watch bots learn to play guitar in real-time
- **Neural Network Visualization**: See the decision-making process inside the AI's brain
- **Genetic Algorithm Training**: Natural selection drives learning without labeled data
- **Interactive Demo Mode**: Watch trained networks perform the solo
- **Progress Tracking**: Fitness graphs and detailed statistics
- **Checkpoint System**: Save and resume training at any generation

## How It Works

### The Challenge

The AI must learn to:
1. **Select the correct string** (1-6) at the right time
2. **Choose the right fret** (0-22) for each note
3. **Play notes with proper timing** (hit notes when they should start)
4. **Complete the entire solo** (34 notes over 14 beats)

### The Training Process

1. **Population**: 8-10 neural networks control different "guitar players"
2. **Evaluation**: Each player attempts the solo, earning fitness based on:
   - Correct notes played (+10 points each)
   - Timing accuracy (+5 bonus per note)
   - Progress through the solo (+50 for completion)
   - Penalties for wrong notes (-2 per mistake)
3. **Selection**: Best performers are kept (elitism)
4. **Reproduction**: Tournament selection + crossover creates offspring
5. **Mutation**: Random weight changes introduce variation
6. **Repeat**: New generation replaces old, process continues

### Neural Network Architecture

```
Input Layer (8 neurons):
  - Current time progress
  - Next note string (target)
  - Next note fret (target)
  - Time until next note
  - Current string position
  - Current fret position
  - Is playing (boolean)
  - Progress through solo

Hidden Layers:
  - Layer 1: 12 neurons (ReLU activation)
  - Layer 2: 12 neurons (ReLU activation)

Output Layer (3 neurons):
  - String selection (-1 to 1 → maps to strings 1-6)
  - Fret selection (-1 to 1 → maps to frets 0-22)
  - Play trigger (-1 to 1 → >0.3 means play note)
```

**Total Parameters**: ~300 trainable weights

## Installation

```bash
# Clone or navigate to the project directory
cd bohemian_rhapsody_ai

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pygame
- Matplotlib

## Usage

### Training

Run the standard training script:

```bash
python main.py
```

**Training Features**:
- 8 players per generation
- 2x simulation speed
- Real-time fretboard visualization for all players
- Progress bars and statistics
- Press `SPACE` to skip to next generation
- Press `ESC` to quit
- Auto-saves checkpoints every 10 generations

### Training with Neural Network Visualization

For a more detailed view of the AI's decision-making:

```bash
python main_visual.py
```

**Enhanced Features**:
- Focus on individual players (press 1-9 keys)
- Neural network visualization showing:
  - Input sensor values
  - Hidden layer activations
  - Output decisions (string, fret, play)
- Real-time statistics for focused player
- Auto-focuses on best performer
- Slower simulation (1x speed) for better observation

### Demo Mode

Watch a trained network perform:

```bash
# Watch the final trained network
python demo.py

# Or specify a specific checkpoint
python demo.py best_network_gen_50.pth
```

**Demo Controls**:
- `ESC`: Quit
- `R`: Reset to beginning
- `SPACE`: Pause/Resume
- Shows note timeline and performance statistics
- Auto-loops when solo completes

## Project Structure

```
bohemian_rhapsody_ai/
├── guitar_solo.py           # Solo note sequence data
├── guitar_player.py         # Player agent with fitness tracking
├── neural_network.py        # PyTorch neural network
├── genetic_algorithm.py     # Evolution algorithm
├── main.py                  # Standard training script
├── main_visual.py           # Enhanced training with NN visualization
├── demo.py                  # Watch trained networks
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── best_network_*.pth      # Saved checkpoints (generated)
└── fitness_history.png     # Performance graph (generated)
```

## Training Results

### Expected Learning Progression

| Generation | Behavior | Fitness Range |
|------------|----------|---------------|
| 0-10 | Random playing, crashes quickly | 0-50 |
| 10-25 | Starting to hit some correct notes | 50-150 |
| 25-50 | Playing multiple notes correctly, improving timing | 150-300 |
| 50-75 | Consistent performance, good accuracy | 300-450 |
| 75+ | Near-optimal play, high completion rate | 450-600+ |

### Typical Training Duration

- **Quick test**: 20-30 generations (~10 minutes)
- **Good performance**: 50 generations (~25 minutes)
- **Optimal**: 75-100 generations (~45 minutes)

Training speed depends on:
- Simulation speed multiplier
- Number of players per generation
- Hardware performance

## Customization

### Modify the Guitar Solo

Edit `guitar_solo.py` to change the note sequence:

```python
GUITAR_SOLO = [
    # (string, fret, duration_beats, timing_beat)
    (3, 5, 0.5, 0.0),    # G string, 5th fret, half beat, at beat 0
    (3, 7, 0.5, 0.5),    # G string, 7th fret, half beat, at beat 0.5
    # ... add more notes
]
```

### Adjust Training Parameters

In `main.py` or `main_visual.py`:

```python
POPULATION_SIZE = 10        # More players = more diversity
SIMULATION_SPEED = 3        # Faster training
MAX_GENERATIONS = 200       # Train longer

# In GeneticAlgorithm initialization:
elite_size = 2              # Keep top 2 performers
mutation_rate = 0.15        # Higher = more exploration
```

### Modify Network Architecture

In `neural_network.py`, change the `__init__` method:

```python
self.fc1 = nn.Linear(8, 16)   # Bigger hidden layer
self.fc2 = nn.Linear(16, 16)  # Another layer
self.fc3 = nn.Linear(16, 3)   # Output layer
```

## How This Differs from Reinforcement Learning

**Genetic Algorithms** (used here):
- ✅ No gradient computation needed
- ✅ Population evaluates in parallel
- ✅ Works with non-differentiable environments
- ✅ Simple to understand and implement
- ❌ Can be slower to converge
- ❌ Less sample efficient

**Reinforcement Learning** (e.g., Q-Learning, PPO):
- ✅ More sample efficient
- ✅ Can learn from individual experiences
- ✅ Better for continuous action spaces
- ❌ Requires careful reward shaping
- ❌ Needs gradient computation
- ❌ More complex to implement

Genetic algorithms are perfect for this guitar learning task because:
1. Fitness is easy to define (correct notes + timing)
2. Multiple players can train simultaneously
3. The environment is deterministic
4. We don't need real-time learning

## Troubleshooting

### "No module named 'pygame'"

Install pygame: `pip install pygame`

### Training is too slow

- Increase `SIMULATION_SPEED` in `main.py`
- Reduce `POPULATION_SIZE`
- Use `main.py` instead of `main_visual.py`

### Networks aren't learning

- Train for more generations (50-100)
- Check mutation rate (0.1-0.2 is good)
- Ensure fitness function rewards correct behavior
- Try increasing population size

### "Network file not found"

Train a network first using `python main.py` before running the demo.

## Extending the Project

Ideas for further development:

1. **Add Audio**: Use `pygame.mixer` to play actual guitar sounds
2. **Multiple Songs**: Train different networks for different solos
3. **Transfer Learning**: Use a trained network as a starting point
4. **Rhythm Variation**: Add swing, syncopation, or different time signatures
5. **Multi-Guitar**: Train multiple guitars to play harmony
6. **Techniques**: Add bends, slides, vibrato, hammer-ons
7. **Real Guitar Input**: Use MIDI guitar to teach by demonstration
8. **Competition Mode**: Let users challenge the AI

## Educational Value

This project demonstrates:

- **Genetic Algorithms**: Evolution through selection, crossover, mutation
- **Neural Networks**: Feed-forward architecture and decision-making
- **Fitness Functions**: Designing objectives for AI agents
- **PyTorch**: Building and manipulating neural networks
- **Pygame**: Real-time visualization and game-like environments
- **Python Best Practices**: Clean code structure and documentation

Perfect for:
- Machine learning students
- Genetic algorithm enthusiasts
- Music + AI projects
- Educational demonstrations

## Credits

Inspired by the neuroevolution projects in the PyTorch tutorial series, particularly the racing car RL implementation.

## License

This project is for educational purposes. Feel free to use, modify, and extend it for learning and teaching!

## Contributing

Found a bug? Have an improvement? Open an issue or submit a pull request!

---

**Happy Training!** Watch your AI learn to shred! 🎸🤖
