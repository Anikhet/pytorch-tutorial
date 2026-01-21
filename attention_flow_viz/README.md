# Attention Flow Visualization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Optional-yellow)

Watch how Vision Transformers "look" at images through beautiful animated attention patterns. See the AI's thought process as flowing heat maps and explore transformer interpretability interactively.

## Learning Objectives

By completing this tutorial, you will learn:

- **Self-Attention Mechanism**: Understand how transformers compute relationships between image patches
- **Vision Transformers (ViT)**: Learn patch embedding, positional encoding, and CLS tokens
- **Multi-Head Attention**: See how different attention heads specialize in different features
- **Attention Rollout**: Compute cumulative information flow across transformer layers
- **Model Interpretability**: Visualize what neural networks "see" and focus on

## Features

- **Attention Heatmaps**: Visualize where the model focuses on images
- **Layer Animation**: Watch attention evolve through 12 transformer layers
- **Head Comparison**: Compare all 12 attention heads side-by-side
- **Attention Rollout**: See accumulated information flow from input to output
- **Entropy Analysis**: Quantitative analysis of attention spread vs. focus

## Quick Start

```bash
# Install dependencies
cd attention_flow_viz
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at http://localhost:8501

## Architecture

### Vision Transformer (ViT-B/16)

```
Input Image (224×224)
        │
        ▼
┌─────────────────────┐
│  Patch Embedding    │   Split into 14×14 = 196 patches of 16×16
│  (16×16 patches)    │   Embed each patch to 768 dimensions
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  + Position Embed   │   Add learnable position embeddings
│  + CLS Token        │   Prepend classification token
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Transformer Block  │ × 12 layers
│  ├─ Multi-Head Attn │   12 heads per layer
│  └─ MLP             │   768 → 3072 → 768
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  CLS Token Output   │   Final representation for classification
└─────────────────────┘
```

| Component | Configuration |
|-----------|--------------|
| Patch size | 16×16 pixels |
| Number of patches | 196 (14×14) |
| Hidden dimension | 768 |
| Number of layers | 12 |
| Attention heads | 12 per layer |
| Total parameters | ~86M |

## Project Structure

```
attention_flow_viz/
├── app.py              # Streamlit application
├── vit_attention.py    # ViT model with attention extraction
├── visualizer.py       # Attention visualization utilities
├── requirements.txt    # Dependencies
└── README.md
```

## How It Works

### Attention Extraction

The model hooks into each transformer layer to capture attention weights:

```python
# Attention shape: [batch, heads, seq_len, seq_len]
# seq_len = 197 (196 patches + 1 CLS token)
attention_weights = model.get_attention_weights(image)
```

### Attention Rollout

Accumulates attention across all layers to show complete information flow:

```python
# Start with identity matrix
rollout = torch.eye(num_patches)

# Multiply through layers
for attention in attention_weights:
    attention = attention + torch.eye(attention.size(-1))  # Residual
    attention = attention / attention.sum(dim=-1, keepdim=True)  # Normalize
    rollout = torch.matmul(attention, rollout)
```

## Hardware Requirements

| Device | Model Load | Inference |
|--------|-----------|-----------|
| CPU | ~5 seconds | ~2 seconds/image |
| M1/M2 Mac | ~3 seconds | ~0.5 seconds/image |
| CUDA GPU | ~2 seconds | ~0.1 seconds/image |

**Requirements**:
- RAM: 4 GB minimum
- Disk: ~500 MB (for ViT-B/16 weights, auto-downloaded)
- Internet: Required for first run (downloads model)

## Understanding the Visualizations

### Heatmap Colors

- **Bright/Hot**: High attention (model focuses here)
- **Dark/Cool**: Low attention (model ignores)

### Layer Patterns

| Layers | Typical Focus |
|--------|--------------|
| 1-4 (Early) | Edges, textures, local features |
| 5-8 (Middle) | Mix of local and global features |
| 9-12 (Late) | Semantic concepts, object regions |

### Head Diversity

Different heads specialize in different features:
- Some focus on edges and boundaries
- Others on textures or colors
- Some on semantic regions (faces, objects)

## Experiments to Try

### 1. Layer Progression

Upload an image and animate through layers. Watch how attention:
- Starts local (edges, textures) in early layers
- Becomes increasingly semantic in later layers
- Converges to key objects/regions

### 2. Head Comparison

View all 12 heads simultaneously:
- Identify which heads focus on background vs foreground
- Find heads specializing in edges vs textures
- Notice some heads are more "spread out" (higher entropy)

### 3. Attention Rollout vs Direct

Compare:
- **Direct attention** (single layer): Shows immediate focus
- **Rollout** (accumulated): Shows complete information flow

### 4. Different Image Types

Try various images to see patterns:
- **Faces**: Attention often focuses on eyes, nose, mouth
- **Objects**: Attention highlights object boundaries
- **Scenes**: Attention spreads across salient regions
- **Abstract**: Watch how the model handles ambiguity

## Troubleshooting

### "Model download failed"

Check internet connection. The ViT-B/16 weights (~350MB) download on first run.

Manual download:
```python
import torchvision.models as models
model = models.vit_b_16(weights='IMAGENET1K_V1')
```

### "CUDA out of memory"

The model runs fine on CPU. Force CPU mode:
```python
device = "cpu"
```

### "Slow inference"

- Use smaller images (model resizes to 224×224)
- Ensure no other GPU processes running
- Consider using fp16 for GPU inference

### "Attention looks uniform"

Some images produce uniform attention:
- Try images with clear subjects/objects
- Ensure good contrast and lighting
- The model was trained on ImageNet - try similar content

## Key Concepts Demonstrated

| Concept | Where to See It |
|---------|-----------------|
| Patch Embedding | `vit_attention.py` - image preprocessing |
| Self-Attention | `vit_attention.py` - attention extraction |
| Multi-Head Attention | Visualizer - head comparison view |
| Attention Rollout | `visualizer.py` - cumulative flow |
| CLS Token | Output aggregation for classification |

## Concepts Covered

### Vision Transformers (ViT)

- Patch embedding (16×16 patches from 224×224 images)
- Self-attention mechanism
- Multi-head attention (12 heads per layer)
- CLS token for classification

### Interpretability Techniques

- **Direct Attention**: Raw attention weights from specific layers/heads
- **Attention Rollout**: Cumulative attention across all layers
- **Entropy Analysis**: Measuring attention spread vs. focus

## Usage Tips

1. **Start with the Layer Animation** to see how attention evolves
2. **Compare heads** to see different "viewpoints" the model uses
3. **Use attention rollout** for the complete information flow picture
4. **Try different images** - faces, objects, scenes, abstract patterns

## References

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., ICLR 2021
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., NeurIPS 2017
- [Quantifying Attention Flow](https://arxiv.org/abs/2005.00928) - Abnar & Zuidema, ACL 2020

## License

MIT License - Part of the PyTorch Tutorial series.
