# Neural Cellular Automata

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Optional-yellow)

Self-organizing patterns that grow from a single cell and can regenerate when damaged.

Based on [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) by Mordvintsev et al.

## Learning Objectives

By completing this tutorial, you will learn:

- **Self-Organization**: Understand how local rules produce global patterns
- **Differentiable Cellular Automata**: Train CA rules with gradient descent
- **Perception Filters**: Use Sobel operators for neighbor communication
- **Pool-Based Sampling**: Maintain diverse training states for stability
- **Damage Recovery**: Train models to regenerate from partial destruction
- **Stochastic Updates**: Use fire rates for training stability

## What is NCA?

Neural Cellular Automata combines:
- **Cellular Automata**: Like Conway's Game of Life, simple local rules create complex patterns
- **Neural Networks**: The rules are learned, not hand-crafted

Each cell:
1. **Perceives** its neighbors using Sobel filters (gradients)
2. **Updates** its state using a tiny neural network
3. **Communicates** only with immediate neighbors

The result: patterns that self-organize, persist, and regenerate.

## Installation

```bash
cd neural_cellular_automata
pip install -r requirements.txt
```

## Quick Start

### Run Demo (Train + Visualize)

```bash
# Full training with heart shape
python demo.py --target heart --steps 2000

# Quick test with circle (500 steps)
python demo.py --quick
```

This creates:
- `outputs/nca_heart_demo.gif` - Growth + regeneration animation
- `outputs/nca_heart_growth.png` - Growth stages grid
- `outputs/nca_heart_regeneration.png` - Damage + repair visualization

### Interactive Web App

```bash
python app.py
# Open http://localhost:7860
```

Features:
- Train on different shapes
- Watch growth in real-time
- Apply damage and observe regeneration
- Generate GIF animations

## Project Structure

```
neural_cellular_automata/
├── nca_model.py      # Core NCA model (perception + update networks)
├── trainer.py        # Training pipeline with pool sampling
├── app.py           # Gradio web interface
├── demo.py          # Command-line demo with visualizations
└── requirements.txt
```

## How It Works

### Architecture

```
State [B, 16, H, W]
    │
    ▼
┌─────────────────┐
│   Perception    │  Sobel-x, Sobel-y, Identity
│   (3x filters)  │  → [B, 48, H, W]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Update MLP     │  Conv1x1 → ReLU → Conv1x1
│  (128 hidden)   │  → [B, 16, H, W]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stochastic     │  Random mask (50% fire rate)
│  Update         │  For training stability
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Alive Mask     │  Cells die if isolated
└────────┬────────┘
         │
         ▼
New State [B, 16, H, W]
```

### Training

- **Pool Sampling**: Maintain a pool of states, sample batches
- **Multi-step Loss**: Run 64-96 steps, compute loss on final RGBA
- **Damage Augmentation**: After warmup, randomly damage samples
- **Overflow Loss**: Penalize values outside [0,1] for stability

## Available Shapes

- `circle` - Red circle
- `square` - Blue square
- `triangle` - Green triangle
- `star` - Yellow 5-pointed star
- `heart` - Pink heart

## Advanced Usage

### Train with Custom Image

```python
from nca_model import NeuralCellularAutomata
from trainer import load_image, NCATrainer

# Load custom target (RGBA PNG)
target = load_image("my_image.png", size=64)

model = NeuralCellularAutomata()
trainer = NCATrainer(model, target, device="cuda")
trainer.train(n_steps=3000)
```

### Generate Animations Programmatically

```python
from demo import create_beautiful_animation

model = ...  # trained model
create_beautiful_animation(
    model,
    size=64,
    n_growth_frames=200,
    output_path="my_animation.gif"
)
```

## GPU Recommendations

| Device | Training Time (2000 steps) |
|--------|---------------------------|
| CPU    | ~10 minutes              |
| M1 Mac | ~3 minutes               |
| RTX 3080 | ~1 minute              |
| H100   | ~30 seconds              |

## References

- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- [Self-Organising Textures](https://distill.pub/selforg/2021/textures/)
- [Original Code](https://github.com/google-research/self-organising-systems)
