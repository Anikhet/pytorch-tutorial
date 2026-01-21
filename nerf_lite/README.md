# NeRF Lite

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Recommended-brightgreen)

Educational implementation of Neural Radiance Fields optimized for fast training on consumer hardware. Learn 3D scene representation with instant-NGP style hash encoding.

## Learning Objectives

By completing this tutorial, you will learn:

- **Neural Radiance Fields**: Understand how to represent 3D scenes as continuous functions F(x,d) → (σ, RGB)
- **Positional Encoding**: Overcome neural network spectral bias with Fourier features
- **Hash Encoding**: Implement instant-NGP style multi-resolution hash tables for 10-100x faster training
- **Volume Rendering**: Master the physics-based rendering equation for alpha compositing
- **Ray Marching**: Generate camera rays and stratified point sampling
- **View Synthesis**: Render photorealistic novel views from arbitrary camera angles

## Features

- **Hash Encoding**: Multi-resolution learnable feature tables for 10-100x faster convergence
- **Volume Rendering**: Physics-based ray marching with proper alpha compositing
- **View-Dependent Color**: Direction-conditioned RGB for specular effects
- **Synthetic Scenes**: Train immediately without downloading external data
- **Blender Support**: Load standard NeRF synthetic datasets
- **PSNR Tracking**: Standard quality metrics during training

## Quick Start

### Installation

```bash
cd nerf_lite
pip install torch torchvision numpy tqdm matplotlib
```

### Train on Synthetic Scene

```bash
# Quick training (50 epochs, ~2-5 minutes on GPU)
python trainer.py --synthetic --epochs 50

# Full training (100 epochs)
python trainer.py --synthetic --epochs 100 --num_views 100

# With positional encoding (classic NeRF)
python trainer.py --synthetic --encoding positional --epochs 200
```

### View Results

Training saves to `pretrained/`:
- `nerf_best.pth` - Best model checkpoint
- `render_epoch_*.png` - Rendered images during training
- `training_curves.png` - Loss and PSNR plots

## Architecture

### NeRF Pipeline

```
Camera Pose → Ray Generation → Point Sampling → Network Query → Volume Rendering → Image
     │              │                │               │                │
     ▼              ▼                ▼               ▼                ▼
  [R|t]        r(t)=o+td        {p₁...pₙ}      (σ,RGB)          C = Σ Tᵢαᵢcᵢ
```

### Network Architecture

```
Position (3D)
    │
    ▼
[Hash Encoding] ──► 16D features
    │
    ▼
[Density Network] ──► sigma (density)
    │                     +
    │               geometry features
    │                     │
    ▼                     ▼
Direction (3D) ──► [Dir Encoding] ──► [Color Network] ──► RGB
```

| Component | Configuration |
|-----------|--------------|
| Density Network | 4 layers × 64 hidden units |
| Color Network | 2 layers × 64 hidden units |
| Geometry Features | 15D intermediate representation |
| Hash Levels | 8 multi-resolution levels |

### "Lite" vs Original NeRF

| Aspect | Original NeRF | NeRF Lite |
|--------|---------------|-----------|
| Encoding | Positional (10 freq) | Hash (8 levels) |
| Hidden dim | 256 | 64 |
| Layers | 8 | 4 |
| Samples/ray | 64 + 128 hierarchical | 64 single-pass |
| Resolution | 800×800 | 64×64 |
| Training | Hours on GPU | Minutes on GPU |

## How It Works

### Volume Rendering Equation

For a ray **r(t) = o + td** from camera origin **o** in direction **d**:

```
C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt

Discrete form:
C = Σᵢ Tᵢ · αᵢ · cᵢ

where:
- αᵢ = 1 - exp(-σᵢ · δᵢ)    # opacity from density
- Tᵢ = Πⱼ₌₁ⁱ⁻¹(1 - αⱼ)      # transmittance (visibility)
- δᵢ = tᵢ₊₁ - tᵢ             # distance between samples
```

### Positional Encoding

Classic Fourier features to overcome spectral bias:

```
γ(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
```

### Hash Encoding (instant-NGP)

1. Create L hash tables at exponentially increasing resolutions
2. For each 3D position, hash to table indices
3. Trilinearly interpolate features from 8 surrounding vertices
4. Concatenate features from all L levels

**Benefits**: Learnable features adapt to scene complexity, 10-100x faster training.

## Project Structure

```
nerf_lite/
├── neural_network.py      # NeRF MLP + positional/hash encoding
├── ray_utils.py           # Camera model, ray generation
├── volume_renderer.py     # Volume rendering equation
├── dataset.py             # Synthetic scenes + Blender loader
├── trainer.py             # Training loop with config
├── pretrained/            # Saved checkpoints
└── README.md
```

### Key Components

| File | Key Classes/Functions |
|------|----------------------|
| `neural_network.py` | `NeRFLiteMLP`, `PositionalEncoding`, `HashEncoding` |
| `ray_utils.py` | `CameraIntrinsics`, `CameraPose`, `generate_rays()` |
| `volume_renderer.py` | `volume_render()`, `compute_psnr()` |
| `dataset.py` | `SyntheticSceneDataset`, `BlenderDataset`, `RayBatchDataset` |
| `trainer.py` | `NeRFTrainer`, `TrainingConfig` |

## Hardware Requirements

| Device | Training (100 epochs) | Render (64×64) |
|--------|----------------------|----------------|
| CPU (Intel i7) | ~60 minutes | ~2 seconds |
| M1/M2 Mac | ~30 minutes | ~1 second |
| RTX 3080 | ~5 minutes | ~0.1 seconds |
| RTX 4090/H100 | ~2 minutes | ~0.05 seconds |

**Minimum Requirements**:
- RAM: 8 GB
- VRAM: 4 GB (if using GPU)
- Disk: 100 MB

## Experiments to Try

### 1. Encoding Comparison

```bash
# Hash encoding (faster)
python trainer.py --synthetic --epochs 100 --encoding hash

# Positional encoding (classic)
python trainer.py --synthetic --epochs 200 --encoding positional
```

**Observe**: Hash reaches ~30 dB PSNR in 100 epochs; positional needs ~300 epochs.

### 2. Scene Complexity

```bash
# Simple (1 sphere)
python trainer.py --synthetic --num_spheres 1 --epochs 50

# Complex (10 spheres)
python trainer.py --synthetic --num_spheres 10 --epochs 150
```

**Watch**: More geometry requires more training capacity.

### 3. Sample Density

```bash
# Sparse (fast, lower quality)
python trainer.py --synthetic --num_samples 32

# Dense (slower, higher quality)
python trainer.py --synthetic --num_samples 128
```

**Balance**: More samples = smoother renders but slower training.

### 4. Network Capacity

```bash
# Tiny network
python trainer.py --synthetic --hidden_dim 32 --num_layers 2

# Larger network
python trainer.py --synthetic --hidden_dim 128 --num_layers 6
```

**Trade-off**: Larger networks capture more detail but train slower.

## Troubleshooting

### "CUDA out of memory"

Reduce batch size or resolution:

```bash
python trainer.py --synthetic --batch_size 512 --image_size 32
```

### "Training loss not decreasing"

- Lower learning rate: `--lr 1e-4`
- Increase samples: `--num_samples 96`
- Check near/far planes match scene bounds

### "Rendered images look noisy"

Increase samples per ray (edit `trainer.py`):

```python
config.num_samples = 128  # Default is 64
```

### "Colors look washed out"

Train longer or increase capacity:

```bash
python trainer.py --synthetic --epochs 200 --hidden_dim 128
```

### Understanding PSNR

| PSNR (dB) | Quality |
|-----------|---------|
| < 20 | Poor - significant artifacts |
| 20-25 | Fair - visible differences |
| 25-30 | Good - minor differences |
| 30-35 | Excellent - nearly indistinguishable |
| > 35 | Outstanding |

## Usage Examples

### Train with Custom Settings

```bash
python trainer.py --synthetic \
    --epochs 100 \
    --num_views 200 \
    --image_size 64 \
    --batch_size 2048 \
    --hidden_dim 64 \
    --encoding hash
```

### Render from Trained Model

```python
from neural_network import NeRFConfig, NeRFLiteMLP
from ray_utils import CameraPose, CameraIntrinsics
import torch

# Load model
checkpoint = torch.load("pretrained/nerf_best.pth")
model = NeRFLiteMLP(NeRFConfig(encoding_type="hash"))
model.load_state_dict(checkpoint["model_state_dict"])

# Create camera
intrinsics = CameraIntrinsics(width=128, height=128, focal_x=100)
pose = CameraPose.from_spherical(radius=4.0, azimuth=0.5, elevation=0.3)
```

## References

- [NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) - Mildenhall et al., ECCV 2020
- [Instant Neural Graphics Primitives](https://nvlabs.github.io/instant-ngp/) - Müller et al., SIGGRAPH 2022
- [Fourier Features Let Networks Learn High Frequency Functions](https://arxiv.org/abs/2006.10739) - Tancik et al., NeurIPS 2020
- [Optical Models for Direct Volume Rendering](https://ieeexplore.ieee.org/document/468400) - Max, 1995

## License

MIT License - Part of the PyTorch Tutorial series.
