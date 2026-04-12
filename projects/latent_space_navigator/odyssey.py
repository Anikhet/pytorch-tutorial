"""
Latent Space Odyssey - Immersive 3D flight through generative latent space.

Fly through VAE/GAN latent space like a spaceship. Watch images morph,
objects transform, and explore the "space of all possible images."

Features:
- 3D latent space navigation with pitch/yaw/roll controls
- Infinite zoom tunnel effect
- Autopilot tours through interesting regions
- Smooth morphing between generated images
- Multiple dataset support (MNIST, Fashion-MNIST, CelebA-style faces)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
from pathlib import Path


class FlightMode(Enum):
    """Flight modes for navigation."""
    MANUAL = "manual"
    AUTOPILOT = "autopilot"
    ORBIT = "orbit"
    WARP = "warp"


@dataclass
class SpaceshipState:
    """State of the spaceship in latent space."""
    position: np.ndarray  # Current position in latent space
    velocity: np.ndarray  # Current velocity vector
    orientation: np.ndarray  # Euler angles [pitch, yaw, roll]
    speed: float = 0.05
    zoom_level: float = 1.0
    flight_mode: FlightMode = FlightMode.MANUAL

    def forward_vector(self) -> np.ndarray:
        """Get the forward direction based on orientation."""
        pitch, yaw, roll = self.orientation
        # Simple 3D direction from Euler angles
        fx = math.cos(yaw) * math.cos(pitch)
        fy = math.sin(pitch)
        fz = math.sin(yaw) * math.cos(pitch)
        return np.array([fx, fy, fz])

    def update(self, dt: float = 1.0):
        """Update position based on velocity."""
        self.position = self.position + self.velocity * dt * self.speed


class OdysseyVAE(nn.Module):
    """
    Enhanced VAE for Latent Space Odyssey with configurable latent dimensions.

    Supports 3D+ latent spaces for spatial navigation while maintaining
    good reconstruction quality.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 28,
        latent_dim: int = 3,
        hidden_dims: List[int] = None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.input_channels = input_channels

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.hidden_dims = hidden_dims

        # Encoder
        encoder_layers = []
        in_ch = input_channels
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_ch, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            in_ch = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate encoder output shape
        self.encoder_out_channels, self.encoder_spatial_size = self._get_encoder_output_shape()
        self.flatten_size = self.encoder_out_channels * self.encoder_spatial_size * self.encoder_spatial_size

        # Latent layers
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)

        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        for i in range(len(hidden_dims_rev) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(hidden_dims_rev[i], hidden_dims_rev[i+1],
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims_rev[i+1]),
                nn.LeakyReLU(0.2),
            ])

        # Final layer
        decoder_layers.extend([
            nn.ConvTranspose2d(hidden_dims_rev[-1], hidden_dims_rev[-1],
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims_rev[-1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims_rev[-1], input_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ])
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _get_encoder_output_shape(self) -> Tuple[int, int]:
        """Calculate the shape after encoder convolutions."""
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            x = self.encoder(x)
            return x.shape[1], x.shape[2]  # channels, spatial_size

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        h = self.decoder_input(z)
        h = h.view(h.size(0), self.encoder_out_channels, self.encoder_spatial_size, self.encoder_spatial_size)
        out = self.decoder(h)

        # Crop/pad to exact input size
        if out.shape[2] != self.input_size or out.shape[3] != self.input_size:
            out = F.interpolate(out, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    @torch.no_grad()
    def sample(self, num_samples: int = 1, device: str = "cpu") -> torch.Tensor:
        """Generate samples from the prior distribution."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def generate_from_position(self, position: np.ndarray) -> torch.Tensor:
        """Generate image from a specific latent position."""
        z = torch.tensor(position, dtype=torch.float32).unsqueeze(0)
        return self.decode(z)


class FlightPath:
    """
    Generate smooth flight paths through latent space.

    Creates bezier curves, spirals, and warp paths for
    autopilot navigation.
    """

    @staticmethod
    def bezier_curve(
        points: List[np.ndarray],
        num_steps: int = 100
    ) -> np.ndarray:
        """Generate a bezier curve through control points."""
        n = len(points) - 1
        t = np.linspace(0, 1, num_steps)

        path = np.zeros((num_steps, len(points[0])))
        for i, p in enumerate(points):
            # Bernstein polynomial
            coeff = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            path += np.outer(coeff, p)

        return path

    @staticmethod
    def spiral_path(
        center: np.ndarray,
        radius_start: float = 2.0,
        radius_end: float = 0.5,
        turns: float = 3.0,
        num_steps: int = 200,
        latent_dim: int = 3
    ) -> np.ndarray:
        """Generate a spiral path converging toward center."""
        t = np.linspace(0, 1, num_steps)
        radius = radius_start + (radius_end - radius_start) * t
        theta = turns * 2 * np.pi * t

        path = np.zeros((num_steps, latent_dim))
        path[:, 0] = center[0] + radius * np.cos(theta)
        path[:, 1] = center[1] + radius * np.sin(theta)
        if latent_dim > 2:
            # Add z-axis motion for 3D
            path[:, 2] = center[2] + (1 - t) * radius_start * 0.5

        return path

    @staticmethod
    def warp_tunnel(
        start: np.ndarray,
        end: np.ndarray,
        num_steps: int = 50,
        warp_intensity: float = 2.0
    ) -> np.ndarray:
        """Generate a warp tunnel path with acceleration."""
        t = np.linspace(0, 1, num_steps)
        # Sigmoid acceleration for warp effect
        warp_t = 1 / (1 + np.exp(-warp_intensity * (t - 0.5) * 10))
        warp_t = (warp_t - warp_t[0]) / (warp_t[-1] - warp_t[0])

        path = np.outer(1 - warp_t, start) + np.outer(warp_t, end)
        return path

    @staticmethod
    def random_walk(
        start: np.ndarray,
        num_steps: int = 500,
        step_size: float = 0.05,
        smoothing: float = 0.9
    ) -> np.ndarray:
        """Generate a smooth random walk through latent space."""
        path = [start.copy()]
        velocity = np.zeros_like(start)

        for _ in range(num_steps - 1):
            # Random acceleration with momentum
            acceleration = np.random.randn(len(start)) * step_size
            velocity = smoothing * velocity + (1 - smoothing) * acceleration
            new_pos = path[-1] + velocity

            # Soft boundary at radius 3
            dist = np.linalg.norm(new_pos)
            if dist > 3:
                new_pos = new_pos * (3 / dist) * 0.95

            path.append(new_pos)

        return np.array(path)


class TunnelEffect:
    """
    Generate trippy infinite tunnel/zoom effects.

    Creates the visual effect of flying through a tunnel
    of generated images.
    """

    def __init__(
        self,
        model: OdysseyVAE,
        num_layers: int = 8,
        layer_scale: float = 1.3,
        device: str = "cpu"
    ):
        self.model = model
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.device = device

    @torch.no_grad()
    def generate_tunnel_frame(
        self,
        center_position: np.ndarray,
        direction: np.ndarray,
        zoom_offset: float = 0.0
    ) -> List[torch.Tensor]:
        """
        Generate layers for tunnel effect.

        Each layer is generated from a position along the flight direction,
        creating a depth illusion.
        """
        layers = []

        for i in range(self.num_layers):
            # Position along the direction
            depth = (i + zoom_offset) % self.num_layers
            offset = direction * depth * 0.3
            position = center_position + offset

            # Generate image at this position
            z = torch.tensor(position, dtype=torch.float32, device=self.device).unsqueeze(0)
            img = self.model.decode(z)
            layers.append(img)

        return layers

    @torch.no_grad()
    def composite_tunnel(
        self,
        layers: List[torch.Tensor],
        output_size: Tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """
        Composite tunnel layers into a single frame.

        Scales and overlays layers to create depth effect.
        """
        from PIL import Image as PILImage

        h, w = output_size
        composite = np.zeros((h, w, 3))

        for i, layer in enumerate(reversed(layers)):
            # Scale factor for depth
            scale = self.layer_scale ** i

            # Convert to numpy
            img = layer[0].cpu().numpy()
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, (1, 2, 0))

            # Resize
            pil_img = PILImage.fromarray((img * 255).astype(np.uint8))
            scaled_size = (int(w * scale), int(h * scale))
            pil_img = pil_img.resize(scaled_size, PILImage.BILINEAR)
            scaled_img = np.array(pil_img) / 255.0

            # Center and blend
            sh, sw = scaled_img.shape[:2]
            y_start = (sh - h) // 2
            x_start = (sw - w) // 2

            if y_start >= 0 and x_start >= 0:
                cropped = scaled_img[y_start:y_start+h, x_start:x_start+w]
                if cropped.shape[:2] == (h, w):
                    # Alpha based on layer depth
                    alpha = 0.7 ** i
                    composite = composite * (1 - alpha) + cropped * alpha

        return np.clip(composite, 0, 1)


class LatentSpaceOdyssey:
    """
    Main controller for the Latent Space Odyssey experience.

    Manages the spaceship state, flight paths, and visualization
    for an immersive journey through latent space.
    """

    def __init__(
        self,
        model: OdysseyVAE,
        device: str = "cpu",
        latent_dim: int = 3
    ):
        self.model = model
        self.device = device
        self.latent_dim = latent_dim

        # Initialize spaceship at origin
        self.ship = SpaceshipState(
            position=np.zeros(latent_dim),
            velocity=np.zeros(latent_dim),
            orientation=np.zeros(3)
        )

        # Flight history for trail visualization
        self.position_history: List[np.ndarray] = []
        self.max_history = 100

        # Autopilot path
        self.autopilot_path: Optional[np.ndarray] = None
        self.autopilot_index: int = 0

        # Tunnel effect
        self.tunnel = TunnelEffect(model, device=device)

        # Points of interest in latent space
        self.waypoints: List[Tuple[str, np.ndarray]] = []

    def set_position(self, position: np.ndarray):
        """Teleport to a specific position."""
        self.ship.position = position.copy()
        self._record_position()

    def _record_position(self):
        """Record current position for trail visualization."""
        self.position_history.append(self.ship.position.copy())
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

    def thrust(self, amount: float = 1.0):
        """Apply forward thrust."""
        direction = self.ship.forward_vector()
        if len(direction) < self.latent_dim:
            # Pad direction if needed
            full_dir = np.zeros(self.latent_dim)
            full_dir[:len(direction)] = direction
            direction = full_dir
        else:
            direction = direction[:self.latent_dim]

        self.ship.velocity = direction * amount
        self.ship.update()
        self._record_position()

    def turn(self, pitch: float = 0, yaw: float = 0, roll: float = 0):
        """Adjust orientation."""
        self.ship.orientation += np.array([pitch, yaw, roll])

    def warp_to(self, target: np.ndarray, steps: int = 30) -> np.ndarray:
        """Initiate warp to target position."""
        path = FlightPath.warp_tunnel(self.ship.position, target, steps)
        return path

    def start_autopilot(self, mode: str = "random"):
        """Start autopilot along a predefined path."""
        self.ship.flight_mode = FlightMode.AUTOPILOT

        if mode == "random":
            self.autopilot_path = FlightPath.random_walk(
                self.ship.position,
                num_steps=500,
                step_size=0.05
            )
        elif mode == "spiral":
            self.autopilot_path = FlightPath.spiral_path(
                np.zeros(self.latent_dim),
                radius_start=2.5,
                latent_dim=self.latent_dim
            )
        elif mode == "tour" and self.waypoints:
            # Visit all waypoints
            points = [self.ship.position] + [wp[1] for wp in self.waypoints]
            self.autopilot_path = FlightPath.bezier_curve(points, num_steps=300)

        self.autopilot_index = 0

    def update_autopilot(self) -> bool:
        """Update position from autopilot. Returns False when path is complete."""
        if self.autopilot_path is None or self.autopilot_index >= len(self.autopilot_path):
            self.ship.flight_mode = FlightMode.MANUAL
            return False

        self.ship.position = self.autopilot_path[self.autopilot_index]
        self.autopilot_index += 1
        self._record_position()
        return True

    @torch.no_grad()
    def get_current_view(self) -> np.ndarray:
        """Generate the current view from spaceship position."""
        z = torch.tensor(self.ship.position, dtype=torch.float32, device=self.device)
        z = z.unsqueeze(0)

        img = self.model.decode(z)
        img = img[0].cpu().numpy()

        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)

        return np.transpose(img, (1, 2, 0))

    @torch.no_grad()
    def get_tunnel_view(self, zoom_offset: float = 0.0) -> np.ndarray:
        """Generate tunnel/infinite zoom view."""
        direction = self.ship.forward_vector()
        if len(direction) < self.latent_dim:
            full_dir = np.zeros(self.latent_dim)
            full_dir[:len(direction)] = direction
            direction = full_dir
        else:
            direction = direction[:self.latent_dim]

        layers = self.tunnel.generate_tunnel_frame(
            self.ship.position,
            direction,
            zoom_offset
        )
        return self.tunnel.composite_tunnel(layers)

    @torch.no_grad()
    def get_neighborhood_grid(self, grid_size: int = 3, spacing: float = 0.5) -> np.ndarray:
        """Generate a grid of images around current position."""
        images = []
        offsets = np.linspace(-spacing, spacing, grid_size)

        for i, dx in enumerate(offsets):
            row = []
            for j, dy in enumerate(offsets):
                pos = self.ship.position.copy()
                pos[0] += dx
                if self.latent_dim > 1:
                    pos[1] += dy

                z = torch.tensor(pos, dtype=torch.float32, device=self.device).unsqueeze(0)
                img = self.model.decode(z)[0].cpu().numpy()

                if img.shape[0] == 1:
                    img = np.repeat(img, 3, axis=0)
                img = np.transpose(img, (1, 2, 0))
                row.append(img)

            images.append(np.concatenate(row, axis=1))

        return np.concatenate(images, axis=0)

    def add_waypoint(self, name: str, position: Optional[np.ndarray] = None):
        """Add a waypoint (bookmark) at current or specified position."""
        pos = position if position is not None else self.ship.position.copy()
        self.waypoints.append((name, pos))

    def discover_interesting_regions(
        self,
        dataloader: DataLoader,
        num_clusters: int = 10
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Analyze dataset to find interesting regions in latent space.

        Computes class centroids and high-variance regions.
        """
        self.model.eval()
        all_codes = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                mu, _ = self.model.encode(images)
                all_codes.append(mu.cpu().numpy())
                all_labels.append(labels.numpy())

        codes = np.concatenate(all_codes)
        labels = np.concatenate(all_labels)

        # Compute class centroids
        regions = []
        for digit in range(10):
            mask = labels == digit
            if mask.sum() > 0:
                centroid = codes[mask].mean(axis=0)
                # Ensure correct dimension
                if len(centroid) < self.latent_dim:
                    full_centroid = np.zeros(self.latent_dim)
                    full_centroid[:len(centroid)] = centroid
                    centroid = full_centroid
                else:
                    centroid = centroid[:self.latent_dim]
                regions.append((f"Digit {digit}", centroid))

        self.waypoints.extend(regions)
        return regions


def create_odyssey_demo():
    """Create an interactive Gradio demo for the Latent Space Odyssey."""
    import gradio as gr
    from PIL import Image

    # Initialize model and odyssey
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Try to load pretrained model or create new one
    model_path = "pretrained/odyssey_vae.pth"
    model = OdysseyVAE(latent_dim=3).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from {model_path}")
    else:
        print("No pretrained model found. Using random initialization.")
        print("Run train_odyssey() to train the model first.")

    model.eval()
    odyssey = LatentSpaceOdyssey(model, device=device, latent_dim=3)

    # State
    zoom_offset = [0.0]

    def update_view(x: float, y: float, z: float, view_mode: str):
        """Update the view based on position and mode."""
        odyssey.set_position(np.array([x, y, z]))

        if view_mode == "Single Image":
            img = odyssey.get_current_view()
        elif view_mode == "Tunnel Effect":
            img = odyssey.get_tunnel_view(zoom_offset[0])
            zoom_offset[0] = (zoom_offset[0] + 0.1) % 8
        else:  # Grid View
            img = odyssey.get_neighborhood_grid(grid_size=3, spacing=0.5)

        # Convert to PIL
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)

    def warp_to_random():
        """Warp to a random position."""
        target = np.random.randn(3) * 2
        return float(target[0]), float(target[1]), float(target[2])

    def start_tour():
        """Start an autopilot tour."""
        odyssey.start_autopilot("random")
        frames = []
        for _ in range(100):
            if not odyssey.update_autopilot():
                break
            img = odyssey.get_current_view()
            img = (img * 255).astype(np.uint8)
            frames.append(Image.fromarray(img))
        return frames

    with gr.Blocks(title="Latent Space Odyssey", theme=gr.themes.Monochrome()) as demo:
        gr.Markdown("""
        # Latent Space Odyssey

        **Fly through the space of all possible images!**

        Navigate through a 3D latent space where every point generates a unique image.
        Watch as forms morph and transform as you travel through this mathematical universe.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Navigation Controls")

                x_slider = gr.Slider(-3, 3, value=0, step=0.1, label="X Axis")
                y_slider = gr.Slider(-3, 3, value=0, step=0.1, label="Y Axis")
                z_slider = gr.Slider(-3, 3, value=0, step=0.1, label="Z Axis")

                view_mode = gr.Radio(
                    ["Single Image", "Tunnel Effect", "Grid View"],
                    value="Single Image",
                    label="View Mode"
                )

                warp_btn = gr.Button("Warp to Random", variant="secondary")
                tour_btn = gr.Button("Start Autopilot Tour", variant="primary")

            with gr.Column(scale=2):
                output_image = gr.Image(label="Current View", height=400)
                tour_gallery = gr.Gallery(label="Tour Frames", columns=10, visible=False)

        # Event handlers
        for slider in [x_slider, y_slider, z_slider]:
            slider.change(
                update_view,
                [x_slider, y_slider, z_slider, view_mode],
                output_image
            )

        view_mode.change(
            update_view,
            [x_slider, y_slider, z_slider, view_mode],
            output_image
        )

        warp_btn.click(
            warp_to_random,
            outputs=[x_slider, y_slider, z_slider]
        )

        tour_btn.click(
            start_tour,
            outputs=tour_gallery
        ).then(lambda: gr.update(visible=True), outputs=tour_gallery)

        gr.Markdown("""
        ---
        ### How It Works

        This demo uses a **Variational Autoencoder (VAE)** with a 3D latent space.
        Each point in the 3D space corresponds to a unique generated image.

        **View Modes:**
        - **Single Image**: See the image at your current position
        - **Tunnel Effect**: Trippy infinite zoom through generated imagery
        - **Grid View**: See a 3x3 neighborhood of nearby images

        **Controls:**
        - Use sliders to navigate the 3D space manually
        - **Warp** jumps to a random location
        - **Autopilot Tour** generates a smooth path through interesting regions
        """)

    return demo


def train_odyssey(
    epochs: int = 50,
    batch_size: int = 128,
    latent_dim: int = 3,
    lr: float = 1e-3,
    beta_warmup: int = 10,
    beta_max: float = 4.0,
    dataset: str = "mnist"
):
    """
    Train the Odyssey VAE model.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        latent_dim: Dimension of latent space (3 recommended for navigation)
        lr: Learning rate
        beta_warmup: Epochs for beta warmup
        beta_max: Maximum beta value for KL loss
        dataset: Dataset to use ("mnist" or "fashion")
    """
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset == "fashion":
        train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    else:
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model
    model = OdysseyVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    save_dir = Path("pretrained")
    save_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Beta warmup
        beta = min(beta_max, beta_max * (epoch + 1) / beta_warmup)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, _ in pbar:
            images = images.to(device)

            recon, mu, log_var = model(images)

            # Loss
            recon_loss = F.binary_cross_entropy(recon, images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item()/len(images):.4f}", beta=f"{beta:.2f}")

        scheduler.step()
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Beta = {beta:.2f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_dir / f"odyssey_vae_epoch{epoch+1}.pth")

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': latent_dim,
    }, save_dir / "odyssey_vae.pth")

    print(f"Training complete! Model saved to {save_dir / 'odyssey_vae.pth'}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Latent Space Odyssey")
    parser.add_argument("--train", action="store_true", help="Train the VAE model")
    parser.add_argument("--demo", action="store_true", help="Run the interactive demo")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--latent-dim", type=int, default=3, help="Latent space dimension")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")

    args = parser.parse_args()

    if args.train:
        train_odyssey(
            epochs=args.epochs,
            latent_dim=args.latent_dim,
            dataset=args.dataset
        )
    elif args.demo:
        demo = create_odyssey_demo()
        demo.launch(server_name="127.0.0.1", server_port=7863)
    else:
        print("Use --train to train the model or --demo to run the interactive demo")
        print("\nExample:")
        print("  python odyssey.py --train --epochs 50")
        print("  python odyssey.py --demo")
