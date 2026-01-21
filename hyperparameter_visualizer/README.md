# Hyperparameter Tuning Visualizer

An interactive tool for experimenting with neural network hyperparameters and watching training progress in real-time.

## Features

- **Real-time visualization** of loss, accuracy, and learning rate curves
- **Interactive controls** for all major hyperparameters
- **Multiple datasets**: Spiral, Circles, Moons, XOR
- **Multiple optimizers**: Adam, SGD, AdamW, RMSprop
- **LR schedulers**: Step, Cosine, Exponential
- **Model configuration**: Hidden size, depth, activation functions

## Quick Start

```bash
cd hyperparameter_visualizer
pip install -r requirements.txt
python demo.py
```

Open http://127.0.0.1:7861 in your browser.

## Interface Overview

### Dataset Settings
- **Dataset**: Choose from spiral, circles, moons, or XOR patterns
- **Samples**: Number of training points (200-2000)
- **Noise**: Amount of noise in the data (0.0-0.5)

### Model Architecture
- **Hidden Size**: Neurons per hidden layer (8-128)
- **Layers**: Number of hidden layers (1-6)
- **Activation**: ReLU, Tanh, LeakyReLU, or GELU

### Optimization
- **Optimizer**: Adam (default), SGD, AdamW, RMSprop
- **Learning Rate**: Initial LR (0.0001-0.5)
- **Batch Size**: Samples per batch (8-256)
- **Epochs**: Training iterations (10-200)

### Regularization
- **Weight Decay**: L2 regularization strength
- **LR Scheduler**: None, Step, Cosine, or Exponential decay

## Experiments to Try

### 1. Learning Rate Impact
```
Dataset: spiral
Optimizer: SGD
Compare: LR=0.001 vs LR=0.1 vs LR=0.5
```
Watch how high learning rates cause oscillation!

### 2. Adam vs SGD
```
Dataset: circles
LR: 0.01
Compare: Adam vs SGD (with momentum)
```
Adam typically converges faster but SGD may generalize better.

### 3. Batch Size Effects
```
Dataset: moons
Optimizer: Adam
Compare: batch=8 vs batch=128 vs batch=256
```
Small batches have more noise but may generalize better.

### 4. Network Depth
```
Dataset: xor
Hidden: 32
Compare: 2 layers vs 5 layers
```
The XOR problem needs non-linearity - depth helps!

### 5. Learning Rate Schedules
```
Dataset: spiral
Optimizer: Adam
LR: 0.01
Compare: none vs cosine vs step
```
Schedules can help fine-tune at the end of training.

## Understanding the Plots

### Loss Curves
- **Blue**: Training loss
- **Red**: Validation loss
- Gap between them indicates overfitting

### Accuracy Curves
- Shows classification accuracy over time
- Validation accuracy is the true measure of performance

### Learning Rate
- Shows how LR changes with schedulers
- Log scale to see small values

### Dataset
- Visualizes the 2D classification problem
- Colors indicate different classes

## File Structure

```
hyperparameter_visualizer/
├── demo.py              # Gradio interface
├── neural_network.py    # MLP models + datasets
├── trainer.py           # Training loop with callbacks
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Key Concepts Demonstrated

### From Notebook 8: Optimization and Tuning

1. **Learning Rate Schedules**
   - Step decay: Reduce LR at fixed intervals
   - Cosine annealing: Smooth decay following cosine curve
   - Exponential: Continuous decay by fixed factor

2. **Optimizer Comparison**
   - SGD: Simple but effective with momentum
   - Adam: Adaptive learning rates, fast convergence
   - AdamW: Proper weight decay implementation

3. **Regularization**
   - Weight decay prevents overfitting
   - Watch the train/val gap!

4. **Batch Size Trade-offs**
   - Small: More gradient noise, slower but may generalize
   - Large: Faster, more stable, but may overfit

## Technical Details

### Networks
- **SimpleMLP**: Standard feedforward network
- **DeepMLP**: Deeper network for complex patterns

### Datasets
- **Spiral**: Classic non-linear problem
- **Circles**: Concentric circles
- **Moons**: Interleaving half-circles
- **XOR**: Classic non-separable problem

### Training
- Cross-entropy loss for classification
- Gradient clipping for stability
- 80/20 train/validation split

## Tips

1. **Start simple**: Begin with defaults, then change one thing at a time
2. **Watch for overfitting**: If val loss increases while train decreases
3. **Learning rate is key**: Usually the most impactful hyperparameter
4. **Be patient**: Some configurations need more epochs

## License

MIT License - Part of the PyTorch Tutorial series.
