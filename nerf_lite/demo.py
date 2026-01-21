"""
Interactive Streamlit demo for NeRF Lite.

This app allows users to:
- Train a NeRF model on synthetic scenes
- Explore novel views by adjusting camera parameters
- Generate 360-degree flythrough videos
- Learn about NeRF concepts through interactive visualizations

Usage:
    streamlit run demo.py
"""

import io
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st

# Set page config first
st.set_page_config(
    page_title="NeRF Lite Explorer",
    page_icon="3D",
    layout="wide"
)


@st.cache_resource
def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load a trained NeRF model."""
    from neural_network import NeRFConfig, NeRFLiteMLP

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to extract config from checkpoint, or use defaults
    model_config = NeRFConfig(encoding_type="hash", hidden_dim=64, num_layers=4)
    model = NeRFLiteMLP(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model


@st.cache_data
def create_synthetic_dataset(_scene_config, num_views: int, image_size: int):
    """Create synthetic dataset (cached)."""
    from dataset import SyntheticScene, SyntheticSceneDataset

    dataset = SyntheticSceneDataset(
        scene_config=_scene_config,
        num_views=num_views,
        image_size=image_size,
        device="cpu"
    )
    return dataset


def render_from_pose(
    model,
    azimuth: float,
    elevation: float,
    distance: float,
    image_size: int,
    num_samples: int,
    near: float,
    far: float,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Render an image from given camera parameters."""
    from ray_utils import CameraIntrinsics, CameraPose, generate_rays, sample_points_along_rays
    from volume_renderer import volume_render

    model_was_training = model.training
    model.eval()

    # Create camera
    intrinsics = CameraIntrinsics(
        width=image_size,
        height=image_size,
        focal_x=image_size * 0.8
    )

    pose = CameraPose.from_spherical(
        radius=distance,
        azimuth=np.radians(azimuth),
        elevation=np.radians(elevation),
        device=device
    )

    # Generate rays
    ray_origins, ray_directions = generate_rays(intrinsics, pose, device)

    # Render in chunks
    all_rgb = []
    all_depth = []
    chunk_size = 1024

    with torch.no_grad():
        for i in range(0, ray_origins.shape[0], chunk_size):
            chunk_origins = ray_origins[i:i + chunk_size]
            chunk_directions = ray_directions[i:i + chunk_size]

            # Sample points
            points, z_vals = sample_points_along_rays(
                chunk_origins, chunk_directions,
                near, far, num_samples,
                stratified=False,
                device=device
            )

            # Forward pass
            N_rays = points.shape[0]
            N_samples = points.shape[1]
            points_flat = points.reshape(-1, 3)
            directions_flat = chunk_directions.unsqueeze(1).expand(-1, N_samples, -1)
            directions_flat = directions_flat.reshape(-1, 3)
            directions_flat = F.normalize(directions_flat, dim=-1)

            density, color = model(points_flat, directions_flat)
            density = density.reshape(N_rays, N_samples, 1)
            color = color.reshape(N_rays, N_samples, 3)

            rgb, depth, _ = volume_render(
                density, color, z_vals, chunk_directions, white_background=True
            )

            all_rgb.append(rgb)
            all_depth.append(depth)

    rgb = torch.cat(all_rgb, dim=0).reshape(image_size, image_size, 3)
    depth = torch.cat(all_depth, dim=0).reshape(image_size, image_size)

    if model_was_training:
        model.train()

    return rgb.cpu().numpy(), depth.cpu().numpy()


def generate_orbit_frames(
    model,
    num_frames: int,
    elevation: float,
    distance: float,
    image_size: int,
    num_samples: int,
    near: float,
    far: float,
    device: str,
    progress_bar
) -> List[np.ndarray]:
    """Generate frames for a 360-degree orbit."""
    frames = []
    for i in range(num_frames):
        azimuth = i * 360.0 / num_frames
        rgb, _ = render_from_pose(
            model, azimuth, elevation, distance,
            image_size, num_samples, near, far, device
        )
        frames.append((rgb * 255).astype(np.uint8))
        progress_bar.progress((i + 1) / num_frames)
    return frames


def train_model_live(
    num_spheres: int,
    num_views: int,
    image_size: int,
    hidden_dim: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    progress_placeholder,
    image_placeholder,
    loss_placeholder
):
    """Train model with live progress updates."""
    from neural_network import NeRFConfig, NeRFLiteMLP
    from dataset import SyntheticScene, SyntheticSceneDataset, RayBatchDataset
    from ray_utils import sample_points_along_rays
    from volume_renderer import volume_render

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataset
    scene_config = SyntheticScene(num_spheres=num_spheres, seed=42)
    train_dataset = SyntheticSceneDataset(
        scene_config=scene_config,
        num_views=num_views,
        image_size=image_size,
        device="cpu"
    )

    ray_dataset = RayBatchDataset(train_dataset, rays_per_sample=batch_size)

    # Create model
    model_config = NeRFConfig(encoding_type="hash", hidden_dim=hidden_dim, num_layers=4)
    model = NeRFLiteMLP(model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    near, far = 2.0, 6.0
    num_samples = 64

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx in range(len(ray_dataset)):
            batch = ray_dataset[batch_idx]
            ray_origins = batch["ray_origins"].to(device)
            ray_directions = batch["ray_directions"].to(device)
            target_rgb = batch["target_rgb"].to(device)

            # Sample points
            points, z_vals = sample_points_along_rays(
                ray_origins, ray_directions,
                near, far, num_samples,
                stratified=True, device=device
            )

            # Forward
            N_rays = points.shape[0]
            N_samples = points.shape[1]
            points_flat = points.reshape(-1, 3)
            directions_flat = ray_directions.unsqueeze(1).expand(-1, N_samples, -1)
            directions_flat = directions_flat.reshape(-1, 3)
            directions_flat = F.normalize(directions_flat, dim=-1)

            density, color = model(points_flat, directions_flat)
            density = density.reshape(N_rays, N_samples, 1)
            color = color.reshape(N_rays, N_samples, 3)

            rgb, _, _ = volume_render(density, color, z_vals, ray_directions)

            loss = F.mse_loss(rgb, target_rgb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        # Update progress
        progress_placeholder.progress((epoch + 1) / num_epochs)

        # Update loss plot
        loss_placeholder.line_chart(losses)

        # Render preview every 10 epochs
        if (epoch + 1) % 10 == 0:
            rgb, _ = render_from_pose(
                model, 45, 30, 4.0, image_size, num_samples, near, far, device
            )
            image_placeholder.image(rgb, caption=f"Epoch {epoch + 1}")

    # Save model
    save_path = Path("pretrained") / "nerf_latest.pth"
    save_path.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config
    }, save_path)

    return model, train_dataset


def main():
    st.title("NeRF Lite Explorer")
    st.markdown("**Neural Radiance Fields for 3D Scene Reconstruction**")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        mode = st.radio("Mode", ["Explore Trained Model", "Train New Model", "Learn About NeRF"])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Device: {device}")

    if mode == "Explore Trained Model":
        explore_mode(device)
    elif mode == "Train New Model":
        train_mode(device)
    else:
        learn_mode()


def explore_mode(device: str):
    """Mode for exploring a trained model."""
    st.header("Explore Trained Model")

    # Check for pretrained model
    model_path = Path("pretrained/nerf_latest.pth")

    if not model_path.exists():
        st.warning("No trained model found. Please train a model first using 'Train New Model' mode.")
        st.code("python trainer.py --synthetic --epochs 50", language="bash")
        return

    # Load model
    model = load_model(str(model_path), device)
    st.success("Model loaded successfully!")

    # Camera controls
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Camera Controls")

        azimuth = st.slider("Azimuth (degrees)", 0, 360, 45)
        elevation = st.slider("Elevation (degrees)", -60, 60, 30)
        distance = st.slider("Distance", 2.0, 6.0, 4.0, 0.1)

        st.subheader("Render Settings")
        image_size = st.slider("Image Size", 32, 128, 64)
        num_samples = st.slider("Samples per Ray", 32, 128, 64)

        near = st.number_input("Near Plane", value=2.0)
        far = st.number_input("Far Plane", value=6.0)

        render_button = st.button("Render View", type="primary")

    with col2:
        st.subheader("Rendered View")

        if render_button:
            with st.spinner("Rendering..."):
                start = time.time()
                rgb, depth = render_from_pose(
                    model, azimuth, elevation, distance,
                    image_size, num_samples, near, far, device
                )
                elapsed = time.time() - start

            col_rgb, col_depth = st.columns(2)
            with col_rgb:
                st.image(rgb, caption="RGB", use_container_width=True)
            with col_depth:
                # Normalize depth for visualization
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
                st.image(depth_norm, caption="Depth", use_container_width=True)

            st.caption(f"Rendered in {elapsed:.2f}s")

    # Flythrough generation
    st.subheader("Generate 360 Flythrough")

    col_fly1, col_fly2 = st.columns(2)

    with col_fly1:
        fly_frames = st.slider("Number of Frames", 8, 60, 24)
        fly_elevation = st.slider("Flythrough Elevation", -60, 60, 20)
        fly_size = st.slider("Flythrough Image Size", 32, 96, 48)

    with col_fly2:
        if st.button("Generate Flythrough", type="secondary"):
            progress_bar = st.progress(0)
            with st.spinner("Generating frames..."):
                frames = generate_orbit_frames(
                    model, fly_frames, fly_elevation, 4.0,
                    fly_size, 48, 2.0, 6.0, device, progress_bar
                )

            # Display as animation using native image cycling
            st.success(f"Generated {len(frames)} frames!")

            # Show as animated sequence
            frame_placeholder = st.empty()
            for i in range(3):  # Loop 3 times
                for frame in frames:
                    frame_placeholder.image(frame, caption="360 Flythrough")
                    time.sleep(0.1)


def train_mode(device: str):
    """Mode for training a new model."""
    st.header("Train New Model")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Training Configuration")

        num_spheres = st.slider("Number of Spheres", 1, 5, 3)
        num_views = st.slider("Training Views", 20, 200, 100)
        image_size = st.slider("Image Size", 32, 128, 64)
        hidden_dim = st.selectbox("Hidden Dimension", [32, 64, 128], index=1)
        num_epochs = st.slider("Epochs", 10, 200, 50)
        batch_size = st.slider("Batch Size", 256, 2048, 1024)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-4, 2e-4, 5e-4, 1e-3],
            value=5e-4,
            format_func=lambda x: f"{x:.0e}"
        )

        train_button = st.button("Start Training", type="primary")

    with col2:
        st.subheader("Training Progress")

        progress_placeholder = st.empty()
        image_placeholder = st.empty()
        loss_placeholder = st.empty()

        if train_button:
            with st.spinner("Training in progress..."):
                model, dataset = train_model_live(
                    num_spheres, num_views, image_size,
                    hidden_dim, num_epochs, batch_size, learning_rate,
                    progress_placeholder, image_placeholder, loss_placeholder
                )
            st.success("Training complete! Model saved to pretrained/nerf_latest.pth")
            st.balloons()


def learn_mode():
    """Educational mode explaining NeRF concepts."""
    st.header("Learn About NeRF")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Volume Rendering",
        "Positional Encoding",
        "Network Architecture"
    ])

    with tab1:
        st.markdown("""
        ## What is NeRF?

        **Neural Radiance Fields (NeRF)** is a method for synthesizing novel views
        of complex scenes from a sparse set of input images.

        ### Key Idea

        NeRF represents a scene as a continuous 5D function:

        ```
        F: (x, y, z, theta, phi) -> (R, G, B, sigma)
        ```

        - **(x, y, z)**: 3D position in space
        - **(theta, phi)**: Viewing direction
        - **(R, G, B)**: Color at that point from that direction
        - **sigma**: Volume density (how "solid" is that point)

        ### How It Works

        1. **Ray Casting**: For each pixel, cast a ray through the scene
        2. **Point Sampling**: Sample points along each ray
        3. **Network Query**: Ask the neural network for density and color at each point
        4. **Volume Rendering**: Integrate colors along the ray using physics-based rendering

        ### Applications

        - **View Synthesis**: Generate new views of a scene
        - **3D Reconstruction**: Build 3D models from photos
        - **VR/AR**: Create immersive experiences from photographs
        - **Visual Effects**: Capture real scenes for digital content
        """)

    with tab2:
        st.markdown("""
        ## Volume Rendering

        The volume rendering equation is the mathematical foundation of NeRF.

        ### The Equation

        For a ray **r(t) = o + td** from camera origin **o** in direction **d**:

        ```
        C(r) = integral from t_n to t_f of T(t) * sigma(r(t)) * c(r(t), d) dt
        ```

        where the **transmittance** T(t) is:

        ```
        T(t) = exp(-integral from t_n to t of sigma(r(s)) ds)
        ```

        ### Intuition

        - **T(t)**: How much light reaches point t (decreases with accumulated density)
        - **sigma**: Volume density - how "thick" is the material at this point
        - **c**: Color - what color is emitted/reflected at this point

        ### Discrete Approximation

        In practice, we sample N points along the ray and compute:

        ```
        C = sum of T_i * alpha_i * c_i for i from 1 to N

        where:
        - alpha_i = 1 - exp(-sigma_i * delta_i)
        - T_i = product of (1 - alpha_j) for j from 1 to i-1
        - delta_i = t_{i+1} - t_i
        ```

        This is like alpha compositing in 2D graphics, extended to 3D!
        """)

        # Interactive visualization
        st.subheader("Interactive: Alpha Compositing")

        density = st.slider("Sample Density (sigma)", 0.0, 10.0, 2.0, 0.1)
        delta = st.slider("Sample Spacing (delta)", 0.01, 0.5, 0.1, 0.01)

        alpha = 1 - np.exp(-density * delta)
        st.metric("Resulting Alpha", f"{alpha:.3f}")

        st.caption(f"alpha = 1 - exp(-{density:.1f} * {delta:.2f}) = {alpha:.3f}")

    with tab3:
        st.markdown("""
        ## Positional Encoding

        Neural networks struggle to learn high-frequency functions. Positional
        encoding maps low-dimensional coordinates to high-dimensional space.

        ### Classic Fourier Features

        ```
        gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p),
                    sin(2^1 * pi * p), cos(2^1 * pi * p),
                    ...,
                    sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]
        ```

        A 3D coordinate (x, y, z) becomes a **6L**-dimensional vector!

        ### Why It Works

        - Networks are biased toward learning **low-frequency** functions
        - Positional encoding provides **high-frequency** basis functions
        - The network only needs to learn the **amplitudes** of each frequency

        ### Hash Encoding (instant-NGP)

        Modern NeRFs use **learnable hash tables** instead:

        1. Create hash tables at multiple resolutions
        2. For each 3D position, hash to table indices
        3. Trilinearly interpolate features
        4. Concatenate features from all resolutions

        **Result**: 10-100x faster training!
        """)

        # Interactive visualization
        st.subheader("Interactive: Positional Encoding")

        x = st.slider("Input value (x)", -1.0, 1.0, 0.5, 0.01)
        num_freq = st.slider("Number of frequencies (L)", 1, 10, 4)

        # Compute encoding
        freqs = 2.0 ** np.arange(num_freq) * np.pi
        encoded = []
        for f in freqs:
            encoded.extend([np.sin(f * x), np.cos(f * x)])

        st.write(f"Encoded from 1D to {len(encoded)}D:")
        st.bar_chart(encoded)

    with tab4:
        st.markdown("""
        ## NeRF Network Architecture

        ### Original NeRF (2020)

        ```
        Position (3D) --> [Positional Encoding] --> [8-layer MLP, 256 units]
                                                            |
                                                            v
                                                    Density (sigma)
                                                            +
                                                    Geometry Features
                                                            |
        Direction (3D) --> [Positional Encoding] ---------->|
                                                            v
                                                    [1-layer MLP] --> RGB Color
        ```

        - Position-only path: Density is view-independent
        - Direction-dependent path: Color varies with viewing angle (specularity)

        ### NeRF Lite (This Implementation)

        Optimizations for faster training:

        | Aspect | Original | Lite |
        |--------|----------|------|
        | Encoding | Positional (10 freq) | Hash (8 levels) |
        | Hidden dim | 256 | 64 |
        | Layers | 8 | 4 |
        | Samples/ray | 64+128 hierarchical | 64 single-pass |
        | Training time | Hours | Minutes |

        ### Key Design Choices

        1. **Separate density/color**: Density is view-independent, color is not
        2. **Skip connections**: Help gradients flow in deep networks
        3. **Hash encoding**: Learnable features adapt to scene complexity
        4. **Coarse-to-fine**: Optional hierarchical sampling for efficiency
        """)


if __name__ == "__main__":
    main()
