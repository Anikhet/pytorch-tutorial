# Sketch-to-Image Diffusion Generator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Optional-yellow)

A CPU-optimized diffusion model that generates images from hand-drawn sketches. Built from scratch for educational purposes, following the concepts from the PyTorch tutorial notebooks.

## Learning Objectives

By completing this tutorial, you will learn:

- **Diffusion Models**: Understand the forward (noising) and reverse (denoising) process
- **U-Net Architecture**: Build a skip-connection architecture for image-to-image tasks
- **DDIM Sampling**: Implement faster deterministic sampling vs stochastic DDPM
- **Conditional Generation**: Guide generation with sketch inputs via concatenation
- **Noise Schedules**: Design variance schedules for stable training
- **Time Embeddings**: Encode timestep information for the denoising network

## Features

- **Custom Diffusion Model**: TinySketchUNet with ~150K parameters
- **CPU Optimized**: Generates images in 1-2 seconds on CPU
- **DDIM Sampling**: 20 steps (vs 1000 for DDPM) for fast inference
- **Gradio Web UI**: Interactive sketch canvas with real-time generation
- **Educational**: Complete implementation from scratch, no black-box libraries

## Quick Start

### 1. Install Dependencies

```bash
cd sketch_to_image_diffusion
pip install -r requirements.txt
```

### 2. Quick Test (Synthetic Data)

Train on synthetic data to verify everything works:

```bash
python train.py --synthetic --epochs 20
```

### 3. Launch Demo

```bash
python demo.py
```

Open http://127.0.0.1:7860 in your browser.

## Training on Real Data

### Download edges2shoes Dataset

The training script will automatically download the dataset:

```bash
python train.py --epochs 100
```

Or manually:

```python
from dataset import download_edges2shoes
download_edges2shoes("data")
```

### Training Options

```bash
python train.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-4 \
    --image-size 32 \
    --timesteps 200
```

### Expected Training Time

| Hardware | Time for 100 epochs |
|----------|---------------------|
| M1/M2 Mac | 2-3 hours |
| Intel i7 | 4-6 hours |
| Intel i5 | 8-12 hours |

## Architecture

### TinySketchUNet

```
Input: [batch, 6, 32, 32]  (3 RGB noisy + 3 sketch)
Output: [batch, 3, 32, 32] (predicted noise)

Encoder:
  Conv(6→32) + GroupNorm + SiLU           [32x32]
  Conv(32→64, stride=2) + GroupNorm + SiLU [16x16]
  Conv(64→128, stride=2) + GroupNorm + SiLU [8x8]

Bottleneck:
  Conv(128→128) + Time Embedding

Decoder:
  Upsample + Skip + Conv(128→64)  [16x16]
  Upsample + Skip + Conv(64→32)   [32x32]
  Conv(32→3)                      [output]
```

### Diffusion Process

1. **Forward (Training)**: Add noise to target images at random timesteps
2. **Reverse (Inference)**: Iteratively denoise from pure noise, conditioned on sketch
3. **DDIM**: Deterministic sampling with fewer steps than DDPM

## File Structure

```
sketch_to_image_diffusion/
├── neural_network.py    # TinySketchUNet architecture
├── diffusion_utils.py   # Noise scheduler, DDIM sampler
├── dataset.py           # Data loading utilities
├── train.py             # Training script
├── demo.py              # Gradio web interface
├── requirements.txt     # Dependencies
├── pretrained/          # Saved models
└── README.md            # This file
```

## How It Works

### Conditioning on Sketches

The model uses **concatenation conditioning**:

```python
# During training and inference:
model_input = torch.cat([noisy_image, sketch], dim=1)  # 6 channels
predicted_noise = model(model_input, timestep)
```

This is simpler and faster than cross-attention conditioning, and works well for spatially-aligned conditions like sketches.

### DDIM vs DDPM

| Method | Steps | Time | Deterministic |
|--------|-------|------|---------------|
| DDPM   | 1000  | ~60s | No |
| DDIM   | 20    | ~2s  | Yes |

DDIM achieves similar quality with 50x fewer steps by using a deterministic update rule.

## Customization

### Adjust Image Size

For higher quality (but slower):

```bash
python train.py --image-size 64 --epochs 200
```

Update `IMAGE_SIZE` in `demo.py` accordingly.

### Adjust Inference Steps

In `demo.py`, modify `DEFAULT_STEPS`:

```python
DEFAULT_STEPS = 30  # Higher quality
```

Or use the slider in the UI.

## Troubleshooting

### "No trained model found"

Train a model first:

```bash
python train.py --synthetic --epochs 50
```

### Out of Memory

Reduce batch size:

```bash
python train.py --batch-size 8
```

### Slow Training

The model is optimized for CPU. For faster training, use:
- Smaller `--epochs`
- `--synthetic` data for testing
- Reduce `--timesteps` to 100

## References

- [DDPM Paper](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [DDIM Paper](https://arxiv.org/abs/2010.02502) - Denoising Diffusion Implicit Models
- [edges2shoes Dataset](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/)

## License

MIT License - Part of the PyTorch Tutorial series.
