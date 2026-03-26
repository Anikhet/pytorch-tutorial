"""
Face Morphing Module for Latent Space Odyssey.

Provides face generation and morphing using a VAE trained on face datasets.
Supports CelebA-style attributes for semantic navigation.

Features:
- Face VAE with higher resolution support (64x64, 128x128)
- Semantic attribute vectors (smile, age, gender, glasses, etc.)
- Smooth morphing between faces
- Face interpolation gallery
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from PIL import Image


@dataclass
class FaceAttributes:
    """Common face attributes for semantic navigation."""
    SMILE = "smile"
    AGE = "age"
    GENDER = "gender"
    GLASSES = "glasses"
    HAIR_COLOR = "hair_color"
    BEARD = "beard"


class FaceVAE(nn.Module):
    """
    VAE optimized for face generation with higher resolution support.

    Architecture designed for 64x64 or 128x128 RGB faces with
    a larger latent space for capturing facial details.
    """

    def __init__(
        self,
        input_size: int = 64,
        latent_dim: int = 128,
        hidden_dims: List[int] = None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        if hidden_dims is None:
            if input_size == 64:
                hidden_dims = [32, 64, 128, 256, 512]
            else:  # 128x128
                hidden_dims = [32, 64, 128, 256, 512, 512]

        self.hidden_dims = hidden_dims

        # Encoder
        encoder_layers = []
        in_ch = 3  # RGB
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_ch, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            in_ch = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate flattened size
        self.flatten_size = self._get_flatten_size()

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

        # Final layer to RGB
        decoder_layers.extend([
            nn.ConvTranspose2d(hidden_dims_rev[-1], hidden_dims_rev[-1],
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims_rev[-1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims_rev[-1], 3, kernel_size=3, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        ])
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _get_flatten_size(self) -> int:
        """Calculate the size after encoder convolutions."""
        with torch.no_grad():
            x = torch.zeros(1, 3, self.input_size, self.input_size)
            x = self.encoder(x)
            return x.numel()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode faces to latent distribution parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to face image."""
        spatial_size = self.input_size // (2 ** len(self.hidden_dims))
        channels = self.flatten_size // (spatial_size * spatial_size)

        h = self.decoder_input(z)
        h = h.view(h.size(0), channels, spatial_size, spatial_size)
        out = self.decoder(h)

        # Ensure correct output size
        if out.shape[2] != self.input_size:
            out = F.interpolate(out, size=(self.input_size, self.input_size),
                               mode='bilinear', align_corners=False)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    @torch.no_grad()
    def generate(self, num_samples: int = 1, device: str = "cpu") -> torch.Tensor:
        """Generate random faces."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        num_steps: int = 10
    ) -> List[torch.Tensor]:
        """Generate interpolation between two latent codes."""
        alphas = torch.linspace(0, 1, num_steps)
        images = []
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            img = self.decode(z)
            images.append(img)
        return images


class SemanticNavigator:
    """
    Navigate face latent space using semantic attributes.

    Learns direction vectors for attributes like smile, age,
    glasses, etc. from labeled data.
    """

    def __init__(self, latent_dim: int = 128):
        self.latent_dim = latent_dim
        self.directions: Dict[str, np.ndarray] = {}

    def compute_direction_from_labels(
        self,
        codes: np.ndarray,
        labels: np.ndarray,
        attribute_name: str
    ) -> np.ndarray:
        """
        Compute semantic direction from binary attribute labels.

        Uses the difference between positive and negative centroids.
        """
        positive_mask = labels == 1
        negative_mask = labels == 0

        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            raise ValueError(f"Need both positive and negative samples for {attribute_name}")

        positive_centroid = codes[positive_mask].mean(axis=0)
        negative_centroid = codes[negative_mask].mean(axis=0)

        direction = positive_centroid - negative_centroid
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        self.directions[attribute_name] = direction
        return direction

    def add_synthetic_direction(self, attribute_name: str):
        """Add a synthetic random direction for demo purposes."""
        direction = np.random.randn(self.latent_dim)
        direction = direction / np.linalg.norm(direction)
        self.directions[attribute_name] = direction

    def apply_direction(
        self,
        z: np.ndarray,
        attribute: str,
        strength: float = 1.0
    ) -> np.ndarray:
        """Apply semantic direction to latent code."""
        if attribute not in self.directions:
            raise ValueError(f"Unknown attribute: {attribute}")

        direction = self.directions[attribute]
        return z + strength * direction

    def get_available_attributes(self) -> List[str]:
        """Get list of available attributes."""
        return list(self.directions.keys())


class FaceMorphingController:
    """
    High-level controller for face morphing operations.

    Combines VAE generation with semantic navigation for
    an interactive face morphing experience.
    """

    def __init__(
        self,
        model: FaceVAE,
        device: str = "cpu"
    ):
        self.model = model
        self.device = device
        self.navigator = SemanticNavigator(latent_dim=model.latent_dim)

        # Initialize synthetic directions for demo
        self._init_demo_directions()

        # Store reference faces
        self.saved_faces: Dict[str, np.ndarray] = {}

    def _init_demo_directions(self):
        """Initialize demo attribute directions."""
        for attr in [FaceAttributes.SMILE, FaceAttributes.AGE,
                    FaceAttributes.GENDER, FaceAttributes.GLASSES]:
            self.navigator.add_synthetic_direction(attr)

    def save_current_face(self, name: str, z: np.ndarray):
        """Save a face latent code for later reference."""
        self.saved_faces[name] = z.copy()

    @torch.no_grad()
    def generate_random_face(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random face. Returns (image, latent_code)."""
        z = torch.randn(1, self.model.latent_dim, device=self.device)
        img = self.model.decode(z)

        # Convert to numpy (H, W, C) format
        img_np = img[0].cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np + 1) / 2  # [-1, 1] -> [0, 1]

        return np.clip(img_np, 0, 1), z[0].cpu().numpy()

    @torch.no_grad()
    def generate_from_latent(self, z: np.ndarray) -> np.ndarray:
        """Generate face from latent code."""
        z_tensor = torch.tensor(z, dtype=torch.float32, device=self.device)
        if z_tensor.dim() == 1:
            z_tensor = z_tensor.unsqueeze(0)

        img = self.model.decode(z_tensor)
        img_np = img[0].cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np + 1) / 2

        return np.clip(img_np, 0, 1)

    def modify_face(
        self,
        z: np.ndarray,
        attribute: str,
        strength: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Modify face using semantic direction."""
        z_modified = self.navigator.apply_direction(z, attribute, strength)
        img = self.generate_from_latent(z_modified)
        return img, z_modified

    @torch.no_grad()
    def morph_faces(
        self,
        z1: np.ndarray,
        z2: np.ndarray,
        num_steps: int = 10,
        method: str = "linear"
    ) -> List[np.ndarray]:
        """Generate morph sequence between two faces."""
        images = []

        for i in range(num_steps):
            alpha = i / (num_steps - 1)

            if method == "spherical":
                # Spherical interpolation
                z1_norm = z1 / (np.linalg.norm(z1) + 1e-8)
                z2_norm = z2 / (np.linalg.norm(z2) + 1e-8)
                omega = np.arccos(np.clip(np.dot(z1_norm, z2_norm), -1, 1))

                if np.abs(omega) < 1e-6:
                    z = (1 - alpha) * z1 + alpha * z2
                else:
                    r1, r2 = np.linalg.norm(z1), np.linalg.norm(z2)
                    r = (1 - alpha) * r1 + alpha * r2
                    direction = (np.sin((1 - alpha) * omega) * z1_norm +
                               np.sin(alpha * omega) * z2_norm) / np.sin(omega)
                    z = r * direction
            else:
                # Linear interpolation
                z = (1 - alpha) * z1 + alpha * z2

            img = self.generate_from_latent(z)
            images.append(img)

        return images

    def create_attribute_grid(
        self,
        z: np.ndarray,
        attribute: str,
        strengths: List[float] = None
    ) -> np.ndarray:
        """Create a grid showing attribute variations."""
        if strengths is None:
            strengths = [-2, -1, 0, 1, 2]

        images = []
        for strength in strengths:
            img, _ = self.modify_face(z, attribute, strength)
            images.append(img)

        # Concatenate horizontally
        return np.concatenate(images, axis=1)


class SyntheticFaceDataset(Dataset):
    """
    Generate synthetic face-like data for testing.

    Creates simple geometric patterns that mimic face structure
    for demo purposes when real face data isn't available.
    """

    def __init__(self, size: int = 5000, image_size: int = 64):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate simple face-like patterns
        img = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        # Random face parameters
        np.random.seed(idx)
        cx, cy = self.image_size // 2, self.image_size // 2

        # Face oval
        y, x = np.ogrid[:self.image_size, :self.image_size]
        face_mask = ((x - cx) / 20) ** 2 + ((y - cy) / 25) ** 2 <= 1
        skin_color = np.random.uniform(0.6, 0.9, 3)
        for c in range(3):
            img[c][face_mask] = skin_color[c]

        # Eyes
        eye_y = cy - 5
        for eye_x in [cx - 8, cx + 8]:
            eye_mask = ((x - eye_x) / 4) ** 2 + ((y - eye_y) / 3) ** 2 <= 1
            for c in range(3):
                img[c][eye_mask] = 0.2

        # Mouth
        mouth_y = cy + 10
        smile = np.random.choice([True, False])
        if smile:
            mouth_mask = ((x - cx) / 8) ** 2 + ((y - mouth_y) / 3) ** 2 <= 1
            mouth_mask &= y > mouth_y
        else:
            mouth_mask = ((x - cx) / 6) ** 2 + ((y - mouth_y) / 2) ** 2 <= 1
        for c in range(3):
            img[c][mouth_mask] = 0.4

        # Random attributes as label
        attributes = {
            'smile': 1 if smile else 0,
            'age': np.random.randint(0, 2),
            'glasses': np.random.randint(0, 2),
        }

        # Convert to tensor
        img = torch.from_numpy(img * 2 - 1)  # Scale to [-1, 1]
        return img, attributes['smile']


def train_face_vae(
    epochs: int = 100,
    batch_size: int = 64,
    latent_dim: int = 128,
    image_size: int = 64,
    lr: float = 1e-4,
    use_synthetic: bool = True
):
    """
    Train the Face VAE model.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        latent_dim: Latent dimension
        image_size: Input image size (64 or 128)
        lr: Learning rate
        use_synthetic: Use synthetic data for demo
    """
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Dataset
    if use_synthetic:
        train_data = SyntheticFaceDataset(size=10000, image_size=image_size)
    else:
        # Would use CelebA here with real face data
        print("Real face dataset not configured. Using synthetic data.")
        train_data = SyntheticFaceDataset(size=10000, image_size=image_size)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = FaceVAE(input_size=image_size, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    save_dir = Path("pretrained")
    save_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        beta = min(1.0, (epoch + 1) / 20)  # Beta warmup

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, _ in pbar:
            images = images.to(device)

            recon, mu, log_var = model(images)

            # Loss with perceptual component
            recon_loss = F.mse_loss(recon, images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item()/len(images):.4f}")

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        if (epoch + 1) % 20 == 0:
            # Save samples
            model.eval()
            with torch.no_grad():
                samples = model.generate(num_samples=16, device=device)
                save_image_grid(samples, save_dir / f"faces_epoch{epoch+1}.png")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, save_dir / f"face_vae_epoch{epoch+1}.pth")

    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': latent_dim,
        'image_size': image_size,
    }, save_dir / "face_vae.pth")

    print(f"Training complete! Model saved to {save_dir / 'face_vae.pth'}")
    return model


def save_image_grid(images: torch.Tensor, path: Path, nrow: int = 4):
    """Save a grid of images."""
    from PIL import Image as PILImage

    n = min(len(images), nrow * nrow)
    images = images[:n]

    # Convert to numpy
    imgs = images.cpu().numpy()
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    imgs = (imgs + 1) / 2  # [-1, 1] -> [0, 1]
    imgs = np.clip(imgs, 0, 1)

    # Create grid
    h, w = imgs.shape[1], imgs.shape[2]
    grid = np.zeros((h * (n // nrow), w * nrow, 3))

    for i, img in enumerate(imgs):
        row = i // nrow
        col = i % nrow
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img

    grid = (grid * 255).astype(np.uint8)
    PILImage.fromarray(grid).save(path)


def create_face_morphing_demo():
    """Create Gradio demo for face morphing."""
    import gradio as gr
    from PIL import Image as PILImage

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load or create model
    model_path = Path("pretrained/face_vae.pth")
    model = FaceVAE(input_size=64, latent_dim=128).to(device)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pretrained face VAE")
    else:
        print("No pretrained model. Using random weights for demo.")

    model.eval()
    controller = FaceMorphingController(model, device)

    # State
    current_z = [None]
    saved_z = [None]

    def generate_new():
        img, z = controller.generate_random_face()
        current_z[0] = z
        return PILImage.fromarray((img * 255).astype(np.uint8))

    def apply_attribute(attr: str, strength: float):
        if current_z[0] is None:
            return generate_new()
        img, z = controller.modify_face(current_z[0], attr, strength)
        current_z[0] = z
        return PILImage.fromarray((img * 255).astype(np.uint8))

    def save_face():
        if current_z[0] is not None:
            saved_z[0] = current_z[0].copy()
        return "Face saved!"

    def morph_to_saved():
        if current_z[0] is None or saved_z[0] is None:
            return []
        images = controller.morph_faces(saved_z[0], current_z[0], num_steps=20)
        return [PILImage.fromarray((img * 255).astype(np.uint8)) for img in images]

    with gr.Blocks(title="Face Morphing Studio") as demo:
        gr.Markdown("""
        # Face Morphing Studio

        Generate and morph faces using semantic attributes.
        Navigate through the latent space of faces!
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gen_btn = gr.Button("Generate New Face", variant="primary")

                gr.Markdown("### Modify Attributes")
                attr_dropdown = gr.Dropdown(
                    choices=["smile", "age", "gender", "glasses"],
                    value="smile",
                    label="Attribute"
                )
                strength_slider = gr.Slider(-3, 3, value=0, step=0.1, label="Strength")
                apply_btn = gr.Button("Apply Attribute")

                gr.Markdown("### Morphing")
                save_btn = gr.Button("Save Current Face")
                save_status = gr.Textbox(label="Status", value="")
                morph_btn = gr.Button("Morph to Saved")

            with gr.Column(scale=2):
                face_display = gr.Image(label="Current Face", height=300)
                morph_gallery = gr.Gallery(label="Morph Sequence", columns=10)

        gen_btn.click(generate_new, outputs=face_display)
        apply_btn.click(apply_attribute, inputs=[attr_dropdown, strength_slider], outputs=face_display)
        save_btn.click(save_face, outputs=save_status)
        morph_btn.click(morph_to_saved, outputs=morph_gallery)

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face Morphing")
    parser.add_argument("--train", action="store_true", help="Train the Face VAE")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    if args.train:
        train_face_vae(epochs=args.epochs)
    elif args.demo:
        demo = create_face_morphing_demo()
        demo.launch(server_name="127.0.0.1", server_port=7864)
    else:
        print("Use --train to train or --demo to run the demo")
