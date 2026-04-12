# Latent Space Navigator & Odyssey

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Optional-yellow)

Interactive exploration of Variational Autoencoder (VAE) latent spaces. Fly through the "space of all possible images" like a spaceship!

## Learning Objectives

By completing this tutorial, you will learn:

- **Variational Autoencoders (VAE)**: Understand probabilistic latent variable models
- **Reparameterization Trick**: Learn the key technique that makes VAEs trainable via backpropagation
- **Latent Space Structure**: Visualize how VAEs organize data in learned representations
- **KL Divergence**: Understand the regularization term that structures the latent space
- **Image Interpolation**: Create smooth transitions between data points
- **Beta-VAE**: Control the trade-off between reconstruction quality and latent regularity

## Features

### Original Navigator (2D)
- **2D Latent Space Map**: Visualize the entire latent space with encoded samples colored by digit class
- **Real-time Generation**: Generate digits by moving through the latent space
- **Digit Navigation**: Jump to any digit's centroid or morph between digits
- **Interpolation**: Create smooth transitions between any two digits

### Latent Space Odyssey (3D) - NEW!
- **3D Navigation**: Fly through latent space with pitch/yaw/roll spaceship controls
- **Infinite Tunnel Effect**: Trippy zoom through generated imagery
- **Autopilot Tours**: Smooth paths through interesting regions
- **Grid View**: See 3x3 neighborhood of images around current position
- **Multiple Datasets**: MNIST, Fashion-MNIST support

### Face Morphing Studio
- **Face Generation**: Generate random faces from VAE latent space
- **Semantic Attributes**: Modify faces with smile, age, gender, glasses controls
- **Face Morphing**: Smooth interpolation between any two faces
- **Attribute Grid**: Visualize attribute variations

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# === Original 2D Navigator ===
python trainer.py        # Train VAE (~5 min on CPU)
python demo.py           # Launch demo at http://127.0.0.1:7862

# === Latent Space Odyssey (3D) ===
python odyssey.py --train --epochs 50   # Train 3D VAE
python odyssey.py --demo                 # Launch at http://127.0.0.1:7863

# === Face Morphing ===
python face_morphing.py --train --epochs 100  # Train Face VAE
python face_morphing.py --demo                 # Launch at http://127.0.0.1:7864
```

## Architecture

### Original VAE (2D)
- **Encoder**: Conv2d layers → 2D latent space (μ, σ)
- **Decoder**: Linear → ConvTranspose2d layers → 28x28 image
- **Parameters**: ~1.3M (CPU-friendly)
- **Latent Dim**: 2 (directly visualizable on 2D map)

### Odyssey VAE (3D)
- **Encoder**: Deep Conv2d layers with BatchNorm → 3D latent space
- **Decoder**: Transposed convolutions → 28x28 image
- **Parameters**: ~2M
- **Latent Dim**: 3 (enables spatial navigation)
- **Special Features**: Tunnel effect compositor, flight path generator

### Face VAE
- **Resolution**: 64x64 RGB images
- **Encoder**: 5-layer ConvNet with BatchNorm
- **Decoder**: Transposed convolutions with Tanh output
- **Latent Dim**: 128 (captures facial details)
- **Semantic Navigation**: Attribute direction vectors

### Training
- **Dataset**: MNIST (60K training, 10K test) / Fashion-MNIST / Synthetic Faces
- **Loss**: Reconstruction (BCE/MSE) + β·KL divergence
- **Beta Warmup**: β from 0→4 over 10 epochs (prevents posterior collapse)
- **Epochs**: 50-100

## UI Components

### Navigation
- **X/Y Sliders**: Move freely through latent space
- **Random Sample**: Jump to a random point

### Digit Controls
- **Jump to Digit**: Teleport to a digit's learned centroid
- **Morph Toward**: Gradually shift current position toward a target digit

### Interpolation
- **Linear**: Straight line path between centroids
- **Spherical**: Geodesic path (often smoother results)

## Files

| File | Description |
|------|-------------|
| `neural_network.py` | Original 2D VAE architecture |
| `trainer.py` | Training loop with beta warmup |
| `latent_utils.py` | Semantic directions, interpolation |
| `demo.py` | Original Gradio interactive UI |
| `odyssey.py` | **NEW** 3D Latent Space Odyssey with spaceship navigation |
| `face_morphing.py` | **NEW** Face VAE with semantic attribute control |
| `pretrained/` | Saved model checkpoints |

## Experiments to Try

### Original Navigator
1. **Digit Morphing**: Start at "3", morph toward "8" - watch the digit transform
2. **Boundary Exploration**: Move between digit clusters to see hybrid forms
3. **Interpolation Comparison**: Compare linear vs spherical interpolation paths
4. **Random Exploration**: Use random sampling to discover unexpected digits

### Latent Space Odyssey
1. **Tunnel Flight**: Enable "Tunnel Effect" mode and slowly move through space
2. **Autopilot Tour**: Start an autopilot tour and watch the smooth morphing
3. **Warp Jumps**: Use "Warp to Random" to teleport to distant regions
4. **Grid Navigation**: Use "Grid View" to see your neighborhood in latent space
5. **Fashion Exploration**: Train on Fashion-MNIST for clothing morphs

### Face Morphing
1. **Attribute Slider**: Generate a face, then adjust smile strength from -3 to +3
2. **Face Morphing**: Save one face, generate another, then morph between them
3. **Age Progression**: Move along the "age" direction to see aging effects
4. **Glasses Toggle**: Add/remove glasses using the semantic direction

## How VAEs Work

A Variational Autoencoder learns to:
1. **Encode** images into a compressed latent representation
2. **Regularize** the latent space to follow a normal distribution
3. **Decode** latent vectors back into images

The 2D latent space lets us visualize how the model organizes digits - similar digits cluster together, and moving through the space produces smooth transitions between digit appearances.

## The Odyssey Concept

The "Latent Space Odyssey" treats the VAE latent space as a navigable universe:

- **Every Point is an Image**: Each coordinate in latent space corresponds to a unique generated image
- **Smooth Transitions**: Moving smoothly through space creates smooth morphing effects
- **Semantic Structure**: Similar concepts cluster together (all "7"s near each other)
- **Infinite Exploration**: The space extends infinitely, with increasingly abstract images at extremes

The 3D navigation metaphor makes exploration intuitive - you're literally flying through the space of all possible images the model can generate.

## Key Concepts Demonstrated

| Concept | Where to See It |
|---------|-----------------|
| **VAE Architecture** | `neural_network.py`, `odyssey.py` |
| **Reparameterization Trick** | `VAE.reparameterize()` method |
| **KL Divergence** | Loss function in training |
| **Beta-VAE** | Beta warmup prevents posterior collapse |
| **Latent Interpolation** | Linear and spherical slerp |
| **Semantic Directions** | Face attribute vectors |
| **Dimensionality Reduction** | 2D/3D latent from high-dim images |

## Hardware Requirements

| Device | Training (50 epochs) | Inference |
|--------|---------------------|-----------|
| CPU (Intel i7) | ~5 minutes | Real-time |
| M1/M2 Mac | ~3 minutes | Real-time |
| CUDA GPU | ~1 minute | Real-time |

**Minimum Requirements**:
- RAM: 4 GB
- Disk: 200 MB (includes MNIST dataset)

## Troubleshooting

### "Posterior collapse" (all outputs look the same)

The KL term dominates, killing the latent space:
- Reduce beta: Start with `beta=0.1`
- Use beta warmup (already implemented)
- Increase reconstruction loss weight

### "Blurry reconstructions"

VAEs naturally produce blurry outputs due to MSE/BCE loss:
- Try MSE loss for sharper results
- Increase latent dimension (more capacity)
- Train longer (more epochs)

### "Latent space has holes"

Regions with no training data produce artifacts:
- Use more training data
- Increase beta for better regularization
- Stay near encoded data points during exploration

### "Demo won't start"

Port conflicts or missing dependencies:
```bash
# Check if port is in use
lsof -i :7862

# Kill existing process
kill -9 <PID>

# Reinstall Gradio
pip install --upgrade gradio
```

## License

MIT License - Part of the PyTorch Tutorial series.
