"""
Interactive Latent Space Navigator.

Gradio interface for exploring VAE latent space with:
- 2D latent space map with sample distribution
- Real-time image generation from position
- Semantic direction sliders (digit morphing)
- Interpolation between points
"""

import os
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from neural_network import VAE
from latent_utils import (
    LatentSpaceAnalyzer,
    linear_interpolation,
    spherical_interpolation,
    encode_dataset
)


MODEL_PATH = "pretrained/vae_mnist.pth"
LATENT_DIM = 2
DEVICE = "cpu"

model = None
analyzer = None
latent_codes = None
labels = None


def load_model():
    """Load pretrained VAE model."""
    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python trainer.py' first."
        )

    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")
    return model


def load_analyzer():
    """Load latent space analyzer with encoded dataset."""
    global analyzer, latent_codes, labels

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    analyzer = LatentSpaceAnalyzer(model, test_loader, DEVICE, max_samples=5000)
    analyzer.analyze()

    latent_codes = analyzer.latent_codes
    labels = analyzer.labels

    return analyzer


@torch.no_grad()
def generate_image(z: np.ndarray) -> Image.Image:
    """Generate image from latent vector."""
    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    image = model.decode(z_tensor)
    image_np = image[0, 0].cpu().numpy()

    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np, mode='L').resize((224, 224), Image.NEAREST)


def create_latent_map(
    current_x: float,
    current_y: float,
    show_samples: bool = True
) -> Image.Image:
    """Create latent space visualization with current position."""
    fig, ax = plt.subplots(figsize=(6, 6))

    if show_samples and latent_codes is not None:
        scatter = ax.scatter(
            latent_codes[:, 0],
            latent_codes[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.3,
            s=10
        )

        for digit in range(10):
            centroid = analyzer.centroids[digit]
            ax.annotate(
                str(digit),
                (centroid[0], centroid[1]),
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                color='white',
                bbox={'facecolor': plt.cm.tab10(digit / 10), 'alpha': 0.8, 'pad': 2}
            )

    ax.scatter([current_x], [current_y], c='red', s=200, marker='x', linewidths=3,
               zorder=10, label='Current Position')

    ax.set_xlabel('z[0]', fontsize=12)
    ax.set_ylabel('z[1]', fontsize=12)
    ax.set_title('Latent Space Map', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()

    return img


def update_from_sliders(x: float, y: float):
    """Update visualization from XY sliders."""
    z = np.array([x, y])
    generated = generate_image(z)
    latent_map = create_latent_map(x, y)
    return generated, latent_map


def move_toward_digit(
    current_x: float,
    current_y: float,
    target_digit: int,
    amount: float
):
    """Move current position toward target digit centroid."""
    current_z = np.array([current_x, current_y])
    new_z = analyzer.move_toward_digit(current_z, int(target_digit), amount)

    new_x, new_y = float(new_z[0]), float(new_z[1])

    generated = generate_image(new_z)
    latent_map = create_latent_map(new_x, new_y)

    return new_x, new_y, generated, latent_map


def jump_to_digit(digit: int):
    """Jump directly to a digit's centroid."""
    centroid = analyzer.centroids[int(digit)]
    x, y = float(centroid[0]), float(centroid[1])

    generated = generate_image(centroid)
    latent_map = create_latent_map(x, y)

    return x, y, generated, latent_map


def generate_interpolation(
    start_digit: int,
    end_digit: int,
    num_steps: int,
    method: str
):
    """Generate interpolation frames between two digits."""
    path = analyzer.interpolate_digits(
        int(start_digit),
        int(end_digit),
        int(num_steps),
        method.lower()
    )

    frames = []
    for z in path:
        img = generate_image(z)
        frames.append(img)

    return frames


def random_sample():
    """Sample a random point from the latent space."""
    z = np.random.randn(2) * 1.5
    x, y = float(z[0]), float(z[1])

    generated = generate_image(z)
    latent_map = create_latent_map(x, y)

    return x, y, generated, latent_map


def build_interface():
    """Build the Gradio interface."""

    with gr.Blocks(
        title="Latent Space Navigator",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # Latent Space Navigator

            **Explore the VAE latent space interactively!**

            Navigate through the 2D latent space and watch digits morph in real-time.
            Each point in the latent space corresponds to a generated digit image.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Navigation Controls")

                x_slider = gr.Slider(
                    minimum=-4.0, maximum=4.0, value=0.0, step=0.1,
                    label="X Position (z[0])"
                )
                y_slider = gr.Slider(
                    minimum=-4.0, maximum=4.0, value=0.0, step=0.1,
                    label="Y Position (z[1])"
                )

                random_btn = gr.Button("Random Sample", variant="secondary")

                gr.Markdown("### Jump to Digit")
                digit_dropdown = gr.Dropdown(
                    choices=[str(i) for i in range(10)],
                    value="0",
                    label="Select Digit"
                )
                jump_btn = gr.Button("Jump to Digit Centroid", variant="primary")

                gr.Markdown("### Morph Toward Digit")
                target_digit = gr.Dropdown(
                    choices=[str(i) for i in range(10)],
                    value="5",
                    label="Target Digit"
                )
                morph_amount = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.1,
                    label="Morph Amount (0=stay, 1=full morph)"
                )
                morph_btn = gr.Button("Morph", variant="secondary")

            with gr.Column(scale=2):
                with gr.Row():
                    latent_map_display = gr.Image(
                        label="Latent Space Map",
                        height=400
                    )
                    generated_display = gr.Image(
                        label="Generated Image",
                        height=400
                    )

        gr.Markdown("---")
        gr.Markdown("### Digit Interpolation")

        with gr.Row():
            with gr.Column(scale=1):
                start_digit = gr.Dropdown(
                    choices=[str(i) for i in range(10)],
                    value="0",
                    label="Start Digit"
                )
                end_digit = gr.Dropdown(
                    choices=[str(i) for i in range(10)],
                    value="9",
                    label="End Digit"
                )
                interp_steps = gr.Slider(
                    minimum=5, maximum=20, value=10, step=1,
                    label="Number of Steps"
                )
                interp_method = gr.Radio(
                    choices=["Linear", "Spherical"],
                    value="Linear",
                    label="Interpolation Method"
                )
                interp_btn = gr.Button("Generate Interpolation", variant="primary")

            with gr.Column(scale=3):
                interp_gallery = gr.Gallery(
                    label="Interpolation Frames",
                    columns=10,
                    rows=1,
                    height=150
                )

        gr.Markdown(
            """
            ---
            ### How It Works

            This demo uses a **Variational Autoencoder (VAE)** trained on MNIST digits.
            The VAE learns a 2D latent space where similar digits cluster together.

            **Features:**
            - **Navigation**: Use sliders to move through the latent space
            - **Digit Jumping**: Jump directly to any digit's learned center point
            - **Morphing**: Gradually transform one digit toward another
            - **Interpolation**: Generate smooth transitions between digits

            **The latent space map shows:**
            - Colored dots: Encoded test samples (color = digit class)
            - Numbers: Centroid of each digit cluster
            - Red X: Your current position
            """
        )

        x_slider.change(
            fn=update_from_sliders,
            inputs=[x_slider, y_slider],
            outputs=[generated_display, latent_map_display]
        )

        y_slider.change(
            fn=update_from_sliders,
            inputs=[x_slider, y_slider],
            outputs=[generated_display, latent_map_display]
        )

        random_btn.click(
            fn=random_sample,
            outputs=[x_slider, y_slider, generated_display, latent_map_display]
        )

        jump_btn.click(
            fn=jump_to_digit,
            inputs=[digit_dropdown],
            outputs=[x_slider, y_slider, generated_display, latent_map_display]
        )

        morph_btn.click(
            fn=move_toward_digit,
            inputs=[x_slider, y_slider, target_digit, morph_amount],
            outputs=[x_slider, y_slider, generated_display, latent_map_display]
        )

        interp_btn.click(
            fn=generate_interpolation,
            inputs=[start_digit, end_digit, interp_steps, interp_method],
            outputs=[interp_gallery]
        )

    return demo


def main():
    """Launch the demo."""
    print("\n" + "=" * 50)
    print("Latent Space Navigator")
    print("=" * 50)

    print("\nLoading model...")
    load_model()

    print("\nAnalyzing latent space...")
    load_analyzer()

    print("\nStarting Gradio interface...")
    demo = build_interface()

    initial_img = generate_image(np.array([0.0, 0.0]))
    initial_map = create_latent_map(0.0, 0.0)

    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True
    )


if __name__ == "__main__":
    main()
