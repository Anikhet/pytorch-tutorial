# Live Training Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Optional-yellow)

A real-time neural network training visualization dashboard built with Streamlit and Plotly.

## Learning Objectives

By completing this tutorial, you will learn:

- **Training Monitoring**: Track loss, accuracy, and gradients in real-time
- **Gradient Flow Analysis**: Detect exploding and vanishing gradients
- **Weight Distributions**: Understand how weights evolve during training
- **Overfitting Detection**: Recognize train/val divergence patterns
- **Learning Rate Effects**: Visualize how LR schedules impact training
- **PyTorch Hooks**: Use forward and backward hooks for model introspection

## Features

- **Real-time loss and accuracy curves** - Watch training progress live
- **Batch-level loss tracking** - See per-batch fluctuations
- **Gradient norm monitoring** - Detect exploding/vanishing gradients
- **Gradient histograms** - Visualize gradient distributions per layer
- **Weight distributions** - Track weight changes during training
- **Model architecture visualization** - See network structure
- **Learning rate schedule** - Monitor LR changes over time
- **Layer-wise statistics** - Detailed stats table

## Quick Start

```bash
cd live_training_dashboard
pip install -r requirements.txt
streamlit run dashboard.py
```

Open http://localhost:8501 in your browser.

## Dashboard Layout

### Sidebar Controls
- **Dataset**: Spiral (2D classification) or MNIST-like
- **Model**: MLP with customizable layers or CNN
- **Training**: Learning rate, batch size, epochs
- **Optimizer**: Adam, SGD, AdamW
- **Scheduler**: None, StepLR, CosineAnnealing

### Main Dashboard
1. **Metrics Bar** - Current epoch, loss, accuracy, gradient norm
2. **Loss Curves** - Train vs validation loss over epochs
3. **Accuracy Curves** - Train vs validation accuracy
4. **Batch Loss** - Per-batch loss with moving average
5. **Gradient Norm** - Track gradient magnitude
6. **Gradient Histograms** - Distribution per layer
7. **Weight Histograms** - Weight distributions
8. **Architecture Diagram** - Visual model structure
9. **Layer Statistics** - Detailed per-layer stats

## File Structure

```
live_training_dashboard/
├── dashboard.py         # Streamlit app
├── neural_network.py    # MonitoredMLP, MonitoredCNN
├── training_monitor.py  # Training loop with metrics
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Key Concepts Demonstrated

### From Notebook 3: Training Your First Model
- Training loops with proper train/val splits
- Loss and accuracy tracking
- Optimizer selection

### From Notebook 27: Monitoring and Observability
- Real-time metric collection
- Gradient monitoring
- Weight distribution tracking
- Training health indicators

## What to Look For

### Healthy Training
- Loss decreasing smoothly
- Train and val loss staying close
- Gradient norms stable (not exploding/vanishing)
- Weights centered around zero with reasonable spread

### Warning Signs

| Symptom | Possible Issue |
|---------|---------------|
| Val loss increasing while train decreases | Overfitting |
| Loss not decreasing | LR too low or architecture issues |
| Loss exploding | LR too high |
| Gradient norms >> 1 | Exploding gradients |
| Gradient norms near 0 | Vanishing gradients |
| Dead neurons (all zeros) | ReLU dying, LR too high |

## Experiments to Try

### 1. Overfitting Detection
```
Settings: Large model, small batch, no regularization
Watch: Train/val gap growing
```

### 2. Learning Rate Comparison
```
Try: 0.001 vs 0.01 vs 0.1
Watch: Convergence speed and stability
```

### 3. Gradient Flow
```
Try: Deep network (128,128,128,128)
Watch: Gradient norms across layers
```

### 4. Scheduler Effects
```
Try: None vs CosineAnnealing
Watch: Learning rate chart and final accuracy
```

## Technical Details

### MonitoredMLP
- Registers forward hooks for activations
- Registers backward hooks for gradients
- Collects statistics after each batch
- Supports customizable architecture

### TrainingMonitor
- Wraps model and optimizer
- Collects detailed metrics every N batches
- Maintains complete training history
- Supports callbacks for live updates

### Dashboard
- Plotly for interactive charts
- Streamlit for reactive UI
- Updates after each training run
- Responsive layout

## Dependencies

- `streamlit>=1.28.0` - Dashboard framework
- `plotly>=5.15.0` - Interactive charts
- `torch>=2.0.0` - Deep learning
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing

## Tips

1. **Start with defaults** - Use Spiral dataset, MLP, Adam
2. **Watch the gap** - Train vs val should stay close
3. **Check gradients** - Norms should be stable
4. **Use histograms** - Spot dead neurons or exploding weights

## License

MIT License - Part of the PyTorch Tutorial series.
