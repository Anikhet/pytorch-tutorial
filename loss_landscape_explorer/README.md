# Loss Landscape Explorer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Optional-yellow)

Interactive 3D visualization of loss landscapes with optimizer trajectory comparison. Watch SGD, Adam, RMSprop, and other optimizers navigate through valleys, saddle points, and local minima with glowing trajectory paths.

## Learning Objectives

By completing this tutorial, you will learn:

- **Loss Landscapes**: Understand how loss functions create complex optimization surfaces
- **Optimizer Behavior**: See how different optimizers navigate valleys, saddle points, and local minima
- **Momentum Effects**: Visualize how momentum helps escape saddle points
- **Adaptive Learning Rates**: Compare Adam/RMSprop's adaptive behavior vs fixed-LR SGD
- **Local vs Global Minima**: Understand why initialization matters in non-convex optimization
- **Gradient Descent Dynamics**: Watch optimization trajectories unfold step-by-step

## Features

- **3D Surface Visualization**: Interactive Plotly surfaces with customizable colorscales
- **Glowing Optimizer Paths**: Multi-layered trajectory rendering with glow effects
- **Side-by-Side Comparison**: Compare up to 6 optimizers simultaneously
- **2D Contour View**: Top-down view for detailed path analysis
- **Animated Playback**: Watch optimization unfold step-by-step
- **Multiple Test Functions**: Rosenbrock, Rastrigin, Himmelblau, Beale, Ackley, Saddle, Goldstein-Price
- **Auto-tuned Learning Rates**: Pre-configured LRs for optimal visualization

## Installation

```bash
cd loss_landscape_explorer
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

### Python API

```python
from landscape import LossLandscape, LandscapeConfig
from optimizer_tracker import OptimizerTracker
from visualizer import LossLandscapeVisualizer

# Compute landscape
landscape = LossLandscape(LandscapeConfig(grid_size=80))
data = landscape.compute_function_landscape("rosenbrock")

# Track optimizer paths
tracker = OptimizerTracker()
paths = tracker.compare_optimizers(
    function_name="rosenbrock",
    optimizer_names=["sgd", "adam", "rmsprop"],
    start_point=(-1.5, 1.5),
    num_steps=150,
)

# Visualize
viz = LossLandscapeVisualizer()
fig = viz.create_single_view(data, list(paths.values()))
fig.show()
```

## Test Functions

| Function | Description | Global Minimum |
|----------|-------------|----------------|
| Rosenbrock | Banana-shaped valley, tests narrow path following | (1, 1) |
| Rastrigin | Highly multimodal, many local minima | (0, 0) |
| Beale | Flat regions with steep walls | (3, 0.5) |
| Himmelblau | Four identical local minima | Multiple |
| Saddle | Simple saddle point at origin | None (saddle) |
| Ackley | Flat outer region with central hole | (0, 0) |
| Goldstein-Price | Complex landscape with flat regions | (0, -1) |

## Supported Optimizers

- **SGD**: Basic stochastic gradient descent
- **SGD + Momentum**: With momentum term (0.9)
- **SGD + Nesterov**: Nesterov accelerated gradient
- **Adam**: Adaptive moment estimation
- **AdamW**: Adam with weight decay
- **RMSprop**: Root mean square propagation
- **Adagrad**: Adaptive gradient algorithm
- **Adadelta**: Extension of Adagrad

## Project Structure

```
loss_landscape_explorer/
├── app.py              # Streamlit web interface
├── landscape.py        # Loss landscape computation
├── optimizer_tracker.py # Optimizer path tracking
├── visualizer.py       # 3D Plotly visualizations
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Visualization Modes

### Combined 3D View
All optimizer trajectories overlaid on a single 3D surface. Best for comparing paths directly.

### Side-by-Side Comparison
Each optimizer gets its own 3D subplot. Best for seeing individual behaviors clearly.

### 2D Contour View
Top-down contour plot with trajectory overlays. Best for detailed path analysis.

### Animated
Step-by-step animation of optimization progress. Best for understanding dynamics.

## Tips

- **Steep functions**: Enable log scale on Z-axis for better visibility
- **Initialization sensitivity**: Try different starting points on Himmelblau
- **Saddle points**: Watch how momentum helps escape the saddle function
- **Local minima**: Compare SGD vs Adam on Rastrigin to see escape behavior
- **Narrow valleys**: Rosenbrock shows why adaptive LR methods excel

## Hardware Requirements

| Device | Landscape Computation | Visualization |
|--------|----------------------|---------------|
| CPU | ~1 second | Real-time |
| M1/M2 Mac | ~0.5 seconds | Real-time |
| CUDA GPU | ~0.1 seconds | Real-time |

**Minimum Requirements**:
- RAM: 2 GB
- No GPU required
- Browser with WebGL support (for Plotly)

## Troubleshooting

### "Streamlit app won't load"

Port conflict or missing dependencies:
```bash
# Check if port is in use
lsof -i :8501

# Run on different port
streamlit run app.py --server.port 8502
```

### "3D visualization is slow/choppy"

Reduce grid resolution:
```python
config = LandscapeConfig(grid_size=50)  # Default is 80
```

### "Optimizer path looks wrong"

Learning rate may be too high/low for the function:
- Try different starting points
- Adjust learning rate in the sidebar
- Use auto-tuned LR settings

### "Plot doesn't render"

WebGL may be disabled:
- Try a different browser (Chrome recommended)
- Enable hardware acceleration in browser settings
- Update graphics drivers

## License

MIT License - Part of the PyTorch Tutorial series.
