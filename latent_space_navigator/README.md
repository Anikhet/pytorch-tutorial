# Latent Space Navigator

Interactive exploration of a Variational Autoencoder (VAE) latent space trained on MNIST digits.

## Features

- **2D Latent Space Map**: Visualize the entire latent space with encoded samples colored by digit class
- **Real-time Generation**: Generate digits by moving through the latent space
- **Digit Navigation**: Jump to any digit's centroid or morph between digits
- **Interpolation**: Create smooth transitions between any two digits

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the VAE (first time only, ~5 min on CPU)
python trainer.py

# Launch the interactive demo
python demo.py
```

Then open http://127.0.0.1:7862 in your browser.

## Architecture

### VAE Structure
- **Encoder**: Conv2d layers → 2D latent space (μ, σ)
- **Decoder**: Linear → ConvTranspose2d layers → 28x28 image
- **Parameters**: ~1.3M (CPU-friendly)
- **Latent Dim**: 2 (directly visualizable)

### Training
- **Dataset**: MNIST (60K training, 10K test)
- **Loss**: Reconstruction (BCE) + β·KL divergence
- **Beta Warmup**: β from 0→4 over 10 epochs (prevents posterior collapse)
- **Epochs**: 50

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
| `neural_network.py` | VAE architecture (Encoder, Decoder, loss) |
| `trainer.py` | Training loop with beta warmup |
| `latent_utils.py` | Semantic directions, interpolation |
| `demo.py` | Gradio interactive UI |
| `pretrained/` | Saved model checkpoint |

## Experiments to Try

1. **Digit Morphing**: Start at "3", morph toward "8" - watch the digit transform
2. **Boundary Exploration**: Move between digit clusters to see hybrid forms
3. **Interpolation Comparison**: Compare linear vs spherical interpolation paths
4. **Random Exploration**: Use random sampling to discover unexpected digits

## How VAEs Work

A Variational Autoencoder learns to:
1. **Encode** images into a compressed latent representation
2. **Regularize** the latent space to follow a normal distribution
3. **Decode** latent vectors back into images

The 2D latent space lets us visualize how the model organizes digits - similar digits cluster together, and moving through the space produces smooth transitions between digit appearances.
