# Quick Start Guide

## Installation (1 minute)

```bash
cd bohemian_rhapsody_ai
pip3 install -r requirements.txt
```

## Training Your First Guitar Bot (5-10 minutes)

### Option 1: Standard Training (Recommended for first time)

```bash
python3 main.py
```

Watch 8 guitar players learn to play simultaneously. Training runs at 2x speed.

**Controls:**
- Press `SPACE` to skip to next generation
- Press `ESC` to quit

**What to expect:**
- Generations 0-10: Random playing, fitness ~20-50
- Generations 10-30: Starting to hit notes, fitness ~50-150
- Generations 30-50: Good accuracy, fitness ~150-300+

Let it train for 30-50 generations for best results (~15-25 minutes).

### Option 2: Visual Training (See the Neural Network Think!)

```bash
python3 main_visual.py
```

Enhanced version showing:
- Live neural network visualization
- Input sensors → Hidden layers → Output decisions
- Detailed statistics for each player

**Extra controls:**
- Press `1-9` to focus on different players
- Auto-focuses on best performer

Runs at 1x speed for better observation.

## Watch Your Trained Bot Perform

After training, watch your bot play:

```bash
python3 demo.py
```

**Demo controls:**
- `ESC`: Quit
- `R`: Reset to start
- `SPACE`: Pause/Resume

Shows:
- Real-time fretboard with target notes (green circles)
- Note timeline (past, present, future notes)
- Performance statistics
- Progress bar

## Quick Test (30 seconds)

Want to see it work immediately? Run a quick 5-generation test:

```bash
python3 main.py  # Let it run for ~2 minutes, press ESC to stop
python3 demo.py  # Watch what it learned
```

Even after just 5-10 generations, you'll see some note-hitting behavior emerging!

## Understanding the Display

### Main Training View (main.py)

```
┌─────────────────────────────────────┐
│ Player 1  Player 2  Player 3  ...  │
│ [fretboard visualization]           │
│ Fitness scores shown above each     │
└─────────────────────────────────────┘
Bottom bar:
- Time progress bar
- Alive count
- Average fitness
- Best player stats
```

### Visual Training View (main_visual.py)

```
┌───────────────┬─────────────────────┐
│   Focused     │   Neural Network    │
│   Player      │   Visualization     │
│ [big view]    │  Input → Hidden →   │
│               │  Output with colors │
│  Statistics   │                     │
└───────────────┴─────────────────────┘
Bottom: Mini views of all players
```

### Demo View (demo.py)

```
┌─────────────────────────────────────┐
│        Guitar Fretboard             │
│  [Your bot playing in real-time]    │
├─────────────────────────────────────┤
│        Note Timeline                │
│  Past → Current → Future notes      │
├─────────────────────────────────────┤
│     Performance Statistics          │
└─────────────────────────────────────┘
```

## Saved Files

After training, you'll find:

- `best_network_gen_10.pth` - Checkpoint at generation 10
- `best_network_gen_20.pth` - Checkpoint at generation 20
- ...
- `best_network_final.pth` - Final best network
- `fitness_history.png` - Graph showing learning progress

To watch a specific checkpoint:
```bash
python3 demo.py best_network_gen_30.pth
```

## Tips for Best Results

1. **Let it train for 50+ generations** - Real learning takes time
2. **Use main.py for faster training** - 2x speed vs visual mode
3. **Check the fitness graph** - Look for upward trend
4. **Try different runs** - Genetic algorithms have randomness
5. **Be patient early on** - First 10 generations are mostly random

## Troubleshooting

### "No module named 'pygame'"
```bash
pip3 install pygame
```

### "Network file not found" when running demo
Train first:
```bash
python3 main.py  # Let it run for at least 10 generations
```

### Training seems stuck at low fitness
- Let it run longer (30-50 generations)
- Try restarting with a new random population
- Genetic algorithms can get stuck in local optima

### Want faster training?
Edit `main.py` and change:
```python
SIMULATION_SPEED = 4  # Instead of 2
POPULATION_SIZE = 6   # Instead of 8
```

## Next Steps

1. ✅ Train for 50 generations
2. ✅ Watch your bot perform in demo mode
3. ✅ Check the fitness graph
4. ✅ Try visual mode to see the neural network
5. ✅ Experiment with different parameters

Ready to see your AI learn to play guitar? Start with:
```bash
python3 main.py
```

Enjoy watching your bot learn! 🎸🤖
