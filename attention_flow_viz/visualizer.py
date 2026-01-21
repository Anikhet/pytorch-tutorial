"""
Attention Visualization Module

Beautiful visualizations for Vision Transformer attention patterns:
- Attention heatmaps overlaid on images
- Animated attention flow across layers
- Head-by-head attention comparison
- Attention rollout visualization
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom

from vit_attention import AttentionMaps


@dataclass
class VisualizerConfig:
    """Configuration for attention visualizer."""

    colormap: str = "magma"
    overlay_alpha: float = 0.6
    interpolation_order: int = 3
    figure_height: int = 600
    figure_width: int = 800
    animation_duration: int = 500


class AttentionVisualizer:
    """Creates beautiful visualizations of ViT attention patterns."""

    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()

    def attention_to_image(
        self,
        attention: torch.Tensor,
        grid_size: int,
        image_size: int = 224
    ) -> np.ndarray:
        """
        Convert attention tensor to image-sized heatmap.

        Args:
            attention: Attention weights [num_patches] or [grid, grid]
            grid_size: Number of patches per side
            image_size: Output image size

        Returns:
            Interpolated attention heatmap [image_size, image_size]
        """
        attn = attention.cpu().numpy()

        if attn.ndim == 1:
            attn = attn.reshape(grid_size, grid_size)

        # Interpolate to image size
        scale = image_size / grid_size
        attn_upscaled = zoom(attn, scale, order=self.config.interpolation_order)

        # Normalize to [0, 1]
        attn_upscaled = (attn_upscaled - attn_upscaled.min()) / (attn_upscaled.max() - attn_upscaled.min() + 1e-8)

        return attn_upscaled

    def create_overlay_image(
        self,
        image: Image.Image,
        attention: np.ndarray,
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Create attention heatmap overlaid on the original image.

        Args:
            image: Original PIL image
            attention: Attention heatmap [H, W] normalized to [0, 1]
            alpha: Overlay transparency

        Returns:
            Blended image with attention overlay [H, W, 3]
        """
        alpha = alpha or self.config.overlay_alpha

        # Resize image to match attention
        img_array = np.array(image.resize((attention.shape[1], attention.shape[0])))
        img_array = img_array.astype(float) / 255.0

        # Apply colormap to attention
        cmap = plt.get_cmap(self.config.colormap)
        attention_colored = cmap(attention)[:, :, :3]

        # Blend
        blended = (1 - alpha) * img_array + alpha * attention_colored

        return np.clip(blended, 0, 1)

    def create_attention_heatmap(
        self,
        image: Image.Image,
        attention_maps: AttentionMaps,
        layer: int,
        head: Optional[int] = None,
        use_rollout: bool = False
    ) -> go.Figure:
        """
        Create interactive attention heatmap visualization.

        Args:
            image: Original image
            attention_maps: Extracted attention maps
            layer: Layer index to visualize
            head: Head index (None for average across heads)
            use_rollout: Use attention rollout instead of single layer

        Returns:
            Plotly figure with attention overlay
        """
        if use_rollout:
            attention = attention_maps.get_attention_rollout(head)
            title = f"Attention Rollout" + (f" (Head {head})" if head is not None else " (All Heads)")
        else:
            if head is not None:
                attention = attention_maps.get_cls_attention(layer, head)
                title = f"Layer {layer + 1}, Head {head + 1}"
            else:
                # Average across heads
                attention = attention_maps.cls_attentions[layer][0].mean(dim=0)
                title = f"Layer {layer + 1} (Average)"

        # Convert to heatmap
        heatmap = self.attention_to_image(
            attention,
            attention_maps.grid_size,
            attention_maps.image_size
        )

        # Create overlay
        overlay = self.create_overlay_image(image, heatmap)

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Image(z=(overlay * 255).astype(np.uint8)))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            width=self.config.figure_width,
            height=self.config.figure_height,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            margin=dict(l=20, r=20, t=50, b=20),
        )

        return fig

    def create_head_comparison(
        self,
        image: Image.Image,
        attention_maps: AttentionMaps,
        layer: int,
        heads_per_row: int = 4
    ) -> go.Figure:
        """
        Create grid comparing attention patterns across all heads.

        Args:
            image: Original image
            attention_maps: Extracted attention maps
            layer: Layer index to visualize
            heads_per_row: Number of heads per row in grid

        Returns:
            Plotly figure with head comparison grid
        """
        num_heads = attention_maps.num_heads
        rows = (num_heads + heads_per_row - 1) // heads_per_row

        fig = make_subplots(
            rows=rows,
            cols=heads_per_row,
            subplot_titles=[f"Head {i + 1}" for i in range(num_heads)],
            horizontal_spacing=0.02,
            vertical_spacing=0.08,
        )

        for head_idx in range(num_heads):
            row = head_idx // heads_per_row + 1
            col = head_idx % heads_per_row + 1

            attention = attention_maps.get_cls_attention(layer, head_idx)
            heatmap = self.attention_to_image(
                attention,
                attention_maps.grid_size,
                attention_maps.image_size
            )
            overlay = self.create_overlay_image(image, heatmap)

            fig.add_trace(
                go.Image(z=(overlay * 255).astype(np.uint8)),
                row=row,
                col=col
            )

        fig.update_layout(
            title=dict(text=f"Layer {layer + 1} - All Attention Heads", font=dict(size=18)),
            height=200 * rows + 100,
            width=200 * heads_per_row + 100,
            showlegend=False,
        )

        # Hide axes for all subplots
        for i in range(1, rows * heads_per_row + 1):
            fig.update_xaxes(showticklabels=False, showgrid=False, row=(i-1)//heads_per_row+1, col=(i-1)%heads_per_row+1)
            fig.update_yaxes(showticklabels=False, showgrid=False, row=(i-1)//heads_per_row+1, col=(i-1)%heads_per_row+1)

        return fig

    def create_layer_animation(
        self,
        image: Image.Image,
        attention_maps: AttentionMaps,
        head: Optional[int] = None
    ) -> go.Figure:
        """
        Create animated visualization showing attention flow through layers.

        Args:
            image: Original image
            attention_maps: Extracted attention maps
            head: Specific head to animate (None for average)

        Returns:
            Animated Plotly figure
        """
        frames = []
        overlays = []

        for layer in range(attention_maps.num_layers):
            if head is not None:
                attention = attention_maps.get_cls_attention(layer, head)
            else:
                attention = attention_maps.cls_attentions[layer][0].mean(dim=0)

            heatmap = self.attention_to_image(
                attention,
                attention_maps.grid_size,
                attention_maps.image_size
            )
            overlay = self.create_overlay_image(image, heatmap)
            overlays.append(overlay)

            frames.append(go.Frame(
                data=[go.Image(z=(overlay * 255).astype(np.uint8))],
                name=str(layer),
                layout=dict(title=dict(text=f"Layer {layer + 1} / {attention_maps.num_layers}"))
            ))

        # Initial frame
        fig = go.Figure(
            data=[go.Image(z=(overlays[0] * 255).astype(np.uint8))],
            frames=frames
        )

        # Animation controls
        fig.update_layout(
            title=dict(
                text="Attention Flow Through Layers" + (f" (Head {head + 1})" if head is not None else ""),
                font=dict(size=18)
            ),
            width=self.config.figure_width,
            height=self.config.figure_height + 100,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.1,
                    xanchor="right",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=self.config.animation_duration, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate"
                                )
                            ]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate"
                                )
                            ]
                        )
                    ]
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            args=[[str(i)], dict(mode="immediate", frame=dict(duration=0))],
                            label=f"L{i + 1}",
                            method="animate"
                        )
                        for i in range(attention_maps.num_layers)
                    ],
                    x=0.1,
                    len=0.8,
                    xanchor="left",
                    y=0,
                    yanchor="top",
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix="Layer: ",
                        visible=True,
                        xanchor="center"
                    ),
                    transition=dict(duration=0),
                )
            ],
            margin=dict(l=20, r=20, t=80, b=80),
        )

        return fig

    def create_attention_flow_graph(
        self,
        attention_maps: AttentionMaps,
        layer: int,
        top_k: int = 5
    ) -> go.Figure:
        """
        Create attention flow graph showing strongest connections between patches.

        Args:
            attention_maps: Extracted attention maps
            layer: Layer to visualize
            top_k: Number of top connections per patch

        Returns:
            Plotly figure with attention flow graph
        """
        grid_size = attention_maps.grid_size
        attention = attention_maps.get_attention_flow(layer).cpu().numpy()

        # Create node positions (patch grid)
        x_nodes = []
        y_nodes = []
        for i in range(grid_size):
            for j in range(grid_size):
                x_nodes.append(j)
                y_nodes.append(grid_size - 1 - i)  # Flip y for display

        # Create edges (top-k strongest attention connections)
        edge_x = []
        edge_y = []
        edge_weights = []

        for patch_idx in range(grid_size * grid_size):
            attn_from_patch = attention[patch_idx]
            top_indices = np.argsort(attn_from_patch)[-top_k:]

            for target_idx in top_indices:
                if patch_idx != target_idx:
                    # Source position
                    src_row, src_col = divmod(patch_idx, grid_size)
                    # Target position
                    tgt_row, tgt_col = divmod(target_idx, grid_size)

                    edge_x.extend([src_col, tgt_col, None])
                    edge_y.extend([grid_size - 1 - src_row, grid_size - 1 - tgt_row, None])
                    edge_weights.append(attn_from_patch[target_idx])

        # Normalize edge weights for opacity
        if edge_weights:
            max_weight = max(edge_weights)
            edge_opacities = [w / max_weight * 0.5 for w in edge_weights]
        else:
            edge_opacities = []

        fig = go.Figure()

        # Add edges
        for i in range(0, len(edge_x), 3):
            if i // 3 < len(edge_opacities):
                fig.add_trace(go.Scatter(
                    x=edge_x[i:i+2],
                    y=edge_y[i:i+2],
                    mode='lines',
                    line=dict(
                        color=f'rgba(255, 100, 100, {edge_opacities[i // 3]})',
                        width=1
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))

        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers',
            marker=dict(
                size=400 // grid_size,
                color=list(range(len(x_nodes))),
                colorscale='Viridis',
                line=dict(width=1, color='white')
            ),
            text=[f"Patch ({i // grid_size}, {i % grid_size})" for i in range(len(x_nodes))],
            hoverinfo='text',
            showlegend=False
        ))

        fig.update_layout(
            title=dict(text=f"Attention Flow - Layer {layer + 1}", font=dict(size=18)),
            width=self.config.figure_width,
            height=self.config.figure_height,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x"),
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=60, b=20),
        )

        return fig

    def create_attention_rollout_comparison(
        self,
        image: Image.Image,
        attention_maps: AttentionMaps
    ) -> go.Figure:
        """
        Compare attention rollout across different heads.

        Args:
            image: Original image
            attention_maps: Extracted attention maps

        Returns:
            Plotly figure comparing rollout across heads
        """
        fig = make_subplots(
            rows=2,
            cols=4,
            subplot_titles=[
                "Original",
                "Rollout (All Heads)",
                "Rollout (Head 1)",
                "Rollout (Head 6)",
                "Layer 1 (Avg)",
                "Layer 6 (Avg)",
                "Layer 9 (Avg)",
                "Layer 12 (Avg)"
            ],
            horizontal_spacing=0.02,
            vertical_spacing=0.1,
        )

        # Original image
        img_resized = image.resize((attention_maps.image_size, attention_maps.image_size))
        fig.add_trace(go.Image(z=np.array(img_resized)), row=1, col=1)

        # Rollout visualizations
        rollout_configs = [
            (None, 1, 2),
            (0, 1, 3),
            (5, 1, 4),
        ]

        for head, row, col in rollout_configs:
            attention = attention_maps.get_attention_rollout(head)
            heatmap = self.attention_to_image(
                attention,
                attention_maps.grid_size,
                attention_maps.image_size
            )
            overlay = self.create_overlay_image(image, heatmap)
            fig.add_trace(go.Image(z=(overlay * 255).astype(np.uint8)), row=row, col=col)

        # Layer-wise attention
        layer_indices = [0, 5, 8, 11]  # Layers 1, 6, 9, 12
        for i, layer_idx in enumerate(layer_indices):
            attention = attention_maps.cls_attentions[layer_idx][0].mean(dim=0)
            heatmap = self.attention_to_image(
                attention,
                attention_maps.grid_size,
                attention_maps.image_size
            )
            overlay = self.create_overlay_image(image, heatmap)
            fig.add_trace(go.Image(z=(overlay * 255).astype(np.uint8)), row=2, col=i+1)

        fig.update_layout(
            title=dict(text="Attention Analysis Overview", font=dict(size=18)),
            height=500,
            width=1000,
            showlegend=False,
        )

        # Hide axes
        for i in range(1, 3):
            for j in range(1, 5):
                fig.update_xaxes(showticklabels=False, showgrid=False, row=i, col=j)
                fig.update_yaxes(showticklabels=False, showgrid=False, row=i, col=j)

        return fig

    def create_attention_statistics(
        self,
        attention_maps: AttentionMaps
    ) -> go.Figure:
        """
        Create visualization of attention statistics across layers.

        Args:
            attention_maps: Extracted attention maps

        Returns:
            Plotly figure with attention statistics
        """
        # Compute statistics
        layers = []
        mean_attentions = []
        entropy_values = []
        max_attentions = []

        for layer in range(attention_maps.num_layers):
            attn = attention_maps.cls_attentions[layer][0].mean(dim=0).cpu().numpy()
            layers.append(layer + 1)
            mean_attentions.append(attn.mean())
            max_attentions.append(attn.max())

            # Compute entropy (measure of attention spread)
            attn_normalized = attn / attn.sum()
            entropy = -np.sum(attn_normalized * np.log(attn_normalized + 1e-10))
            entropy_values.append(entropy)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Mean Attention", "Max Attention", "Attention Entropy"]
        )

        fig.add_trace(
            go.Scatter(x=layers, y=mean_attentions, mode='lines+markers',
                      marker=dict(size=8, color='blue'), name="Mean"),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=layers, y=max_attentions, mode='lines+markers',
                      marker=dict(size=8, color='red'), name="Max"),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=layers, y=entropy_values, mode='lines+markers',
                      marker=dict(size=8, color='green'), name="Entropy"),
            row=1, col=3
        )

        fig.update_layout(
            title=dict(text="Attention Statistics by Layer", font=dict(size=18)),
            height=350,
            width=1000,
            showlegend=False,
        )

        fig.update_xaxes(title_text="Layer", row=1, col=1)
        fig.update_xaxes(title_text="Layer", row=1, col=2)
        fig.update_xaxes(title_text="Layer", row=1, col=3)

        return fig
