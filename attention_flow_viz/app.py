"""
Attention Flow Visualization - Streamlit App

Watch how Vision Transformers "look" at images through animated attention heads.
See the AI's thought process as flowing particles and heat maps.
Explore transformer interpretability interactively.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io

from vit_attention import ViTAttentionModel, AttentionMaps, create_synthetic_image
from visualizer import AttentionVisualizer, VisualizerConfig


@st.cache_resource
def load_model():
    """Load the ViT model (cached)."""
    return ViTAttentionModel(pretrained=True)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Attention Flow Visualization",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔮 Attention Flow Visualization")
    st.markdown(
        """
        Watch how Vision Transformers perceive images through their attention patterns.
        See the AI's "thought process" as beautiful flowing heat maps.
        """
    )

    # Load model
    with st.spinner("Loading Vision Transformer..."):
        model = load_model()

    st.success(f"✓ Loaded ViT-B/16 ({model.num_layers} layers, {model.num_heads} heads)")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Image source selection
        st.subheader("📷 Image Source")
        image_source = st.radio(
            "Select image source",
            options=["upload", "synthetic", "url"],
            format_func=lambda x: {
                "upload": "Upload Image",
                "synthetic": "Synthetic Sample",
                "url": "Image URL"
            }[x]
        )

        image = None

        if image_source == "upload":
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "webp"],
                help="Upload any image to analyze"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")

        elif image_source == "synthetic":
            synthetic_type = st.selectbox(
                "Synthetic pattern",
                options=["circles", "gradient", "stripes", "random"],
            )
            image = create_synthetic_test_image(synthetic_type)

        else:  # URL
            url = st.text_input(
                "Image URL",
                value="",
                placeholder="https://example.com/image.jpg"
            )
            if url:
                try:
                    import urllib.request
                    with urllib.request.urlopen(url, timeout=10) as response:
                        image_data = response.read()
                        image = Image.open(io.BytesIO(image_data)).convert("RGB")
                except Exception as e:
                    st.error(f"Failed to load image: {e}")

        st.divider()

        # Visualization settings
        st.subheader("🎨 Visualization")

        colormap = st.selectbox(
            "Color scheme",
            options=["magma", "viridis", "plasma", "inferno", "hot", "cool"],
            index=0,
            help="Colormap for attention heatmaps"
        )

        overlay_alpha = st.slider(
            "Overlay intensity",
            min_value=0.2,
            max_value=0.9,
            value=0.6,
            step=0.1,
            help="How strongly to blend attention over image"
        )

        st.divider()

        # Layer and head selection
        st.subheader("🔍 Exploration")

        selected_layer = st.slider(
            "Layer",
            min_value=1,
            max_value=model.num_layers,
            value=model.num_layers,
            help="Which transformer layer to visualize"
        )

        head_mode = st.radio(
            "Attention head",
            options=["average", "specific"],
            format_func=lambda x: "Average (all heads)" if x == "average" else "Specific head"
        )

        selected_head = None
        if head_mode == "specific":
            selected_head = st.slider(
                "Head number",
                min_value=1,
                max_value=model.num_heads,
                value=1,
            ) - 1

    # Main content
    if image is None:
        st.info("👆 Upload an image or select a sample to get started!")
        st.markdown(
            """
            ### What you'll see:
            - **Attention Heatmaps**: Where the model focuses on the image
            - **Layer Animation**: How attention evolves through 12 transformer layers
            - **Head Comparison**: Different "viewpoints" the model uses
            - **Attention Rollout**: Accumulated attention from input to output

            ### Try it with:
            - Photos of animals (dogs, cats, birds)
            - Objects (cars, furniture, food)
            - Scenes (landscapes, buildings)
            - Abstract patterns (to see unexpected attention)
            """
        )
        return

    # Process image
    with st.spinner("Analyzing attention patterns..."):
        pred_class, confidence, attention_maps = model.predict(image)

    # Create visualizer with config
    viz_config = VisualizerConfig(
        colormap=colormap,
        overlay_alpha=overlay_alpha
    )
    visualizer = AttentionVisualizer(viz_config)

    # Display prediction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"### 🎯 Prediction: **{pred_class}**")
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")

    st.divider()

    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔥 Attention Heatmap",
        "🎬 Layer Animation",
        "👁️ Head Comparison",
        "🌊 Attention Rollout",
        "📊 Statistics"
    ])

    with tab1:
        st.subheader("Attention Heatmap")
        st.markdown(
            """
            This shows where the Vision Transformer "looks" when making its prediction.
            Bright regions receive more attention from the CLS (classification) token.
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown(f"**Layer {selected_layer} Attention**")
            fig = visualizer.create_attention_heatmap(
                image,
                attention_maps,
                layer=selected_layer - 1,
                head=selected_head
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Attention Flow Through Layers")
        st.markdown(
            """
            Watch how attention patterns evolve as information flows through the 12 transformer layers.
            Early layers often focus on edges and textures, while later layers attend to semantic concepts.
            """
        )

        animation_head = selected_head if head_mode == "specific" else None
        fig = visualizer.create_layer_animation(image, attention_maps, head=animation_head)
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 Use the Play button or drag the slider to explore different layers!")

    with tab3:
        st.subheader("Attention Head Comparison")
        st.markdown(
            """
            Each attention head acts as a different "viewpoint" looking at the image.
            Some heads focus on textures, others on shapes, colors, or semantic regions.
            """
        )

        display_layer = st.select_slider(
            "Select layer to compare heads",
            options=list(range(1, model.num_layers + 1)),
            value=model.num_layers
        )

        fig = visualizer.create_head_comparison(
            image,
            attention_maps,
            layer=display_layer - 1
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Attention Rollout")
        st.markdown(
            """
            Attention rollout shows how information flows from input patches to the final classification.
            This accumulates attention across all layers, revealing the complete path of information flow.
            """
        )

        fig = visualizer.create_attention_rollout_comparison(image, attention_maps)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            **Reading the visualization:**
            - **Rollout (All Heads)**: Combined attention flow from all heads
            - **Rollout (Head N)**: Flow through a specific attention head
            - **Layer N**: Direct attention at specific layers (1=early, 12=late)
            """
        )

    with tab5:
        st.subheader("Attention Statistics")
        st.markdown(
            """
            Quantitative analysis of attention patterns across layers.
            """
        )

        fig = visualizer.create_attention_statistics(attention_maps)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Attention Flow Network**")
            flow_layer = st.slider(
                "Layer for flow visualization",
                min_value=1,
                max_value=model.num_layers,
                value=6,
                key="flow_layer"
            )
            fig = visualizer.create_attention_flow_graph(
                attention_maps,
                layer=flow_layer - 1,
                top_k=3
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Understanding the Metrics**")
            st.markdown(
                """
                - **Mean Attention**: Average attention weight across patches
                - **Max Attention**: Strongest attention to any single patch
                - **Entropy**: Measure of attention spread (higher = more distributed)

                **Patterns to look for:**
                - Later layers often have higher max attention (more focused)
                - Earlier layers may have higher entropy (looking everywhere)
                - Peaks indicate important processing stages
                """
            )

    # Footer
    st.divider()
    with st.expander("📚 About Vision Transformer Attention"):
        st.markdown(
            """
            ### How Vision Transformers Work

            **Patch Embedding:**
            The image is divided into 16x16 patches, each embedded as a token.
            For a 224x224 image, this creates 196 patch tokens.

            **Self-Attention:**
            Each layer performs self-attention where every patch can "look at" every other patch.
            The attention weights determine how much information flows between patches.

            **CLS Token:**
            A special [CLS] token aggregates information for classification.
            The attention FROM [CLS] TO patches shows what the model deems important.

            **Multiple Heads:**
            Each layer has 12 attention heads, each potentially focusing on different features:
            - Some heads may focus on edges
            - Others on textures or colors
            - Some on semantic regions (faces, objects)

            **Interpretability:**
            - **Attention Heatmaps**: Direct visualization of attention weights
            - **Attention Rollout**: Accumulated flow through all layers
            - **GradCAM**: Gradient-based importance (not shown here)

            ### Model Details
            - **Architecture**: ViT-B/16 (Base variant, 16x16 patches)
            - **Layers**: 12 transformer blocks
            - **Heads**: 12 attention heads per layer
            - **Hidden Dim**: 768
            - **Parameters**: ~86M
            - **Training**: ImageNet-1K (1000 classes)
            """
        )


def create_synthetic_test_image(pattern: str) -> Image.Image:
    """Create synthetic test images for demonstration."""
    size = 224
    img = np.zeros((size, size, 3), dtype=np.uint8)

    if pattern == "circles":
        # Concentric circles
        y, x = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2
        for r in range(20, 110, 20):
            mask = np.abs(np.sqrt((x - cx)**2 + (y - cy)**2) - r) < 5
            color = np.array([
                int(255 * (r / 110)),
                int(100 + 100 * np.sin(r / 20)),
                int(255 - 200 * (r / 110))
            ])
            img[mask] = color

    elif pattern == "gradient":
        # Radial gradient with center spot
        y, x = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        img[:, :, 0] = (255 * (1 - dist / (size * 0.7))).clip(0, 255).astype(np.uint8)
        img[:, :, 1] = (200 * np.sin(dist / 30)).clip(0, 255).astype(np.uint8)
        img[:, :, 2] = (255 * (dist / (size * 0.7))).clip(0, 255).astype(np.uint8)

        # Add center circle
        center_mask = dist < 30
        img[center_mask] = [255, 255, 100]

    elif pattern == "stripes":
        # Diagonal stripes with varying colors
        for i in range(size):
            for j in range(size):
                stripe = ((i + j) // 20) % 4
                if stripe == 0:
                    img[i, j] = [200, 50, 50]
                elif stripe == 1:
                    img[i, j] = [50, 200, 50]
                elif stripe == 2:
                    img[i, j] = [50, 50, 200]
                else:
                    img[i, j] = [200, 200, 50]

    else:  # random
        np.random.seed(42)
        # Random colored rectangles
        img[:] = [100, 100, 120]
        for _ in range(10):
            x1, y1 = np.random.randint(0, size - 40, 2)
            w, h = np.random.randint(30, 80, 2)
            color = np.random.randint(50, 255, 3)
            img[y1:y1+h, x1:x1+w] = color

    return Image.fromarray(img)


if __name__ == "__main__":
    main()
