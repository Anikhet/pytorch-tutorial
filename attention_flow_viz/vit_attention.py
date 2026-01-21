"""
Vision Transformer Attention Extraction Module

Provides utilities to load pretrained Vision Transformers and extract
attention weights from all layers and heads for visualization.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import numpy as np
from PIL import Image


@dataclass
class AttentionMaps:
    """Container for extracted attention maps from a ViT forward pass."""

    attentions: list[torch.Tensor]  # List of [batch, heads, tokens, tokens]
    cls_attentions: list[torch.Tensor]  # CLS token attention per layer
    patch_size: int = 16
    image_size: int = 224
    num_patches: int = field(init=False)
    grid_size: int = field(init=False)

    def __post_init__(self):
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size ** 2

    @property
    def num_layers(self) -> int:
        return len(self.attentions)

    @property
    def num_heads(self) -> int:
        return self.attentions[0].shape[1] if self.attentions else 0

    def get_attention(self, layer: int, head: int) -> torch.Tensor:
        """Get attention matrix for specific layer and head."""
        return self.attentions[layer][0, head]

    def get_cls_attention(self, layer: int, head: int) -> torch.Tensor:
        """Get CLS token's attention to patches for specific layer/head."""
        return self.cls_attentions[layer][0, head]

    def get_attention_rollout(self, head: Optional[int] = None) -> torch.Tensor:
        """
        Compute attention rollout across all layers.

        Attention rollout recursively multiplies attention matrices
        to show how information flows from input patches to the CLS token.
        """
        device = self.attentions[0].device

        # Start with identity matrix
        rollout = torch.eye(
            self.attentions[0].shape[-1],
            device=device,
            dtype=self.attentions[0].dtype
        )

        for attn in self.attentions:
            if head is not None:
                attn_head = attn[0, head]
            else:
                # Average across heads
                attn_head = attn[0].mean(dim=0)

            # Add residual connection (identity)
            attn_with_residual = 0.5 * attn_head + 0.5 * torch.eye(
                attn_head.shape[-1],
                device=device,
                dtype=attn_head.dtype
            )

            # Normalize
            attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)

            # Accumulate
            rollout = rollout @ attn_with_residual

        # Return CLS token attention to patches (excluding CLS token itself)
        return rollout[0, 1:]

    def get_attention_flow(self, layer: int) -> torch.Tensor:
        """
        Get attention flow matrix showing how patches attend to each other.
        Returns averaged attention across heads, excluding CLS token.
        """
        attn = self.attentions[layer][0].mean(dim=0)  # Average heads
        # Exclude CLS token (first row and column)
        return attn[1:, 1:]


class AttentionExtractor:
    """Extracts attention weights from Vision Transformer models."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_maps: list[torch.Tensor] = []
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        for name, module in self.model.named_modules():
            if 'self_attention' in name and hasattr(module, 'out_proj'):
                # This is the attention layer in torchvision ViT
                parent_name = '.'.join(name.split('.')[:-1])
                parent = dict(self.model.named_modules())[parent_name]

                # We need to hook the encoder block, not just attention
                if hasattr(parent, 'self_attention'):
                    hook = parent.register_forward_hook(self._create_hook())
                    self._hooks.append(hook)

    def _create_hook(self):
        """Create a hook function that captures attention weights."""
        def hook(module, input_tensor, output):
            # For torchvision ViT, we need to compute attention ourselves
            # since they don't expose it directly
            pass
        return hook

    def clear(self):
        """Clear stored attention maps."""
        self.attention_maps = []

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class ViTAttentionModel:
    """
    Vision Transformer wrapper that extracts attention weights during inference.

    Uses a custom forward pass to capture attention from each transformer block.
    """

    def __init__(
        self,
        model_name: str = "vit_b_16",
        pretrained: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model_name = model_name
        self.patch_size = 16
        self.image_size = 224

        # Load pretrained ViT
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.model = vit_b_16(weights=weights)
            self.categories = weights.meta["categories"]
        else:
            self.model = vit_b_16()
            self.categories = None

        self.model = self.model.to(device)
        self.model.train(False)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Get model architecture info
        self.num_heads = self.model.encoder.layers[0].self_attention.num_heads
        self.num_layers = len(self.model.encoder.layers)
        self.hidden_dim = self.model.hidden_dim

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image for the model."""
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def forward_with_attention(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, AttentionMaps]:
        """
        Forward pass that returns both predictions and attention maps.

        Args:
            x: Preprocessed image tensor [1, 3, 224, 224]

        Returns:
            logits: Classification logits
            attention_maps: AttentionMaps container with all attention weights
        """
        attentions = []
        cls_attentions = []

        # Patch embedding
        x = self.model.conv_proj(x)  # [B, hidden_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]

        # Add class token
        batch_size = x.shape[0]
        cls_token = self.model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, 1 + num_patches, hidden_dim]

        # Add positional embedding
        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)

        # Process through transformer blocks
        for layer in self.model.encoder.layers:
            # Self-attention with explicit attention computation
            attn_output, attn_weights = self._compute_attention(
                layer.self_attention,
                layer.ln_1(x)
            )
            x = x + attn_output

            # MLP
            x = x + layer.mlp(layer.ln_2(x))

            # Store attention weights
            attentions.append(attn_weights)
            cls_attentions.append(attn_weights[:, :, 0, 1:])  # CLS to patches

        # Final layer norm and classification
        x = self.model.encoder.ln(x)
        x = x[:, 0]  # Take CLS token
        logits = self.model.heads(x)

        attention_maps = AttentionMaps(
            attentions=attentions,
            cls_attentions=cls_attentions,
            patch_size=self.patch_size,
            image_size=self.image_size
        )

        return logits, attention_maps

    def _compute_attention(
        self,
        self_attention: nn.Module,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute self-attention and return both output and attention weights.
        """
        batch_size, seq_len, embed_dim = x.shape
        num_heads = self_attention.num_heads
        head_dim = embed_dim // num_heads

        # Compute Q, K, V
        qkv = self_attention.in_proj_weight @ x.transpose(-2, -1)
        qkv = qkv.transpose(-2, -1) + self_attention.in_proj_bias
        qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention weights
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self_attention.dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        attn_output = self_attention.out_proj(attn_output)

        return attn_output, attn_weights

    def predict(self, image: Image.Image) -> tuple[str, float, AttentionMaps]:
        """
        Get prediction and attention maps for an image.

        Returns:
            class_name: Predicted class name
            confidence: Prediction confidence
            attention_maps: Extracted attention maps
        """
        x = self.preprocess(image)
        logits, attention_maps = self.forward_with_attention(x)

        probs = torch.softmax(logits, dim=-1)
        confidence, pred_idx = probs.max(dim=-1)

        class_name = self.categories[pred_idx.item()] if self.categories else f"Class {pred_idx.item()}"

        return class_name, confidence.item(), attention_maps


def load_sample_images() -> dict[str, Image.Image]:
    """Load sample images for demonstration."""
    import urllib.request
    import io

    samples = {}

    # Sample image URLs (from Wikimedia Commons - free to use)
    sample_urls = {
        "cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "dog": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
    }

    for name, url in sample_urls.items():
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                image_data = response.read()
                samples[name] = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception:
            # Create synthetic sample if download fails
            samples[name] = create_synthetic_image(name)

    return samples


def create_synthetic_image(name: str) -> Image.Image:
    """Create a simple synthetic test image."""
    np.random.seed(hash(name) % 2**32)

    # Create gradient background with shapes
    img = np.zeros((224, 224, 3), dtype=np.uint8)

    # Gradient background
    for i in range(224):
        for j in range(224):
            img[i, j] = [
                int(100 + 100 * (i / 224)),
                int(100 + 100 * (j / 224)),
                int(150)
            ]

    # Add some circles
    for _ in range(5):
        cx, cy = np.random.randint(30, 194, 2)
        r = np.random.randint(15, 40)
        color = np.random.randint(50, 255, 3)

        y, x = np.ogrid[:224, :224]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        img[mask] = color

    return Image.fromarray(img)
