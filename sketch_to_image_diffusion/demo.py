"""
Gradio Demo for Sketch-to-Image Diffusion.

Launch with: python demo.py

Features:
- Sketchpad for drawing
- Quality slider (DDIM steps)
- Real-time generation
- Example gallery
"""

from pathlib import Path
import numpy as np
import torch
import gradio as gr

from neural_network import TinySketchUNet
from diffusion_utils import NoiseScheduler, ddim_sample, preprocess_sketch


# Configuration
MODEL_PATH = Path(__file__).parent / "pretrained" / "sketch2image_final.pth"
IMAGE_SIZE = 32
NUM_TIMESTEPS = 200
DEFAULT_STEPS = 20


def load_model():
    """Load the trained model."""
    model = TinySketchUNet()

    if MODEL_PATH.exists():
        print(f"Loading model from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    else:
        print(f"Warning: No trained model found at {MODEL_PATH}")
        print("The model will generate random outputs.")
        print("Train a model first with: python train.py --synthetic --epochs 50")

    model.set_mode_to_inference()
    return model


# Monkey-patch for cleaner code
TinySketchUNet.set_mode_to_inference = lambda self: self.eval()


# Global model and scheduler
print("Initializing Sketch-to-Image Diffusion Demo...")
MODEL = load_model()
SCHEDULER = NoiseScheduler(num_timesteps=NUM_TIMESTEPS, device="cpu")


def generate_from_sketch(sketch_input, num_steps: int = DEFAULT_STEPS):
    """
    Generate an image from a user-drawn sketch.

    Args:
        sketch_input: Input from Gradio Sketchpad (dict or numpy array)
        num_steps: Number of DDIM denoising steps

    Returns:
        Generated image as numpy array (RGB, 0-255)
    """
    if sketch_input is None:
        return None

    try:
        # Preprocess sketch
        sketch_tensor = preprocess_sketch(sketch_input, size=IMAGE_SIZE, device="cpu")
        sketch_tensor = sketch_tensor.unsqueeze(0)  # Add batch dimension

        # Generate image
        with torch.inference_mode():
            generated = ddim_sample(
                MODEL,
                sketch_tensor,
                SCHEDULER,
                num_inference_steps=int(num_steps),
                show_progress=False
            )

        # Convert to displayable format
        output = generated[0].permute(1, 2, 0).numpy()  # [H, W, C]
        output = ((output + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        # Upscale for better display (32x32 is tiny)
        from PIL import Image
        img = Image.fromarray(output)
        img = img.resize((256, 256), Image.Resampling.NEAREST)
        output = np.array(img)

        return output

    except Exception as e:
        print(f"Error generating image: {e}")
        return None


# Build the Gradio interface
def build_interface():
    """Build and return the Gradio interface."""

    with gr.Blocks(
        title="Sketch to Image - Diffusion Demo",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # Sketch-to-Image Generator

            **Draw a sketch and watch diffusion magic turn it into an image!**

            This demo uses a CPU-optimized diffusion model trained on edge-to-image pairs.
            The model learns to denoise images conditioned on sketch inputs.

            **How to use:**
            1. Draw a sketch in the canvas (shoe outlines work best with edges2shoes training)
            2. Adjust the quality slider (more steps = better quality, slower)
            3. Click "Generate Image"
            """
        )

        with gr.Row():
            with gr.Column():
                # Sketch input
                sketch_input = gr.Sketchpad(
                    label="Draw your sketch here",
                    brush=gr.Brush(colors=["#000000"], default_size=3),
                    height=300,
                    width=300,
                    type="numpy"
                )

                # Controls
                with gr.Row():
                    clear_btn = gr.Button("Clear", size="sm")
                    generate_btn = gr.Button("Generate Image", variant="primary", size="lg")

                # Quality slider
                quality_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=DEFAULT_STEPS,
                    step=5,
                    label="Quality (DDIM Steps)",
                    info="Higher = better quality, slower generation"
                )

            with gr.Column():
                # Output image
                output_image = gr.Image(
                    label="Generated Image",
                    height=300,
                    width=300,
                    type="numpy"
                )

                # Generation info
                gr.Markdown(
                    """
                    **Tips:**
                    - Simple outlines work best
                    - Try drawing shoe shapes (if trained on edges2shoes)
                    - 10-20 steps is good for quick previews
                    - 30-50 steps for higher quality
                    """
                )

        # Event handlers
        generate_btn.click(
            fn=generate_from_sketch,
            inputs=[sketch_input, quality_slider],
            outputs=output_image
        )

        clear_btn.click(
            fn=lambda: None,
            outputs=sketch_input
        )

        # Footer
        gr.Markdown(
            """
            ---
            **Model Details:**
            - Architecture: TinySketchUNet (~150K parameters)
            - Resolution: 32x32 (upscaled to 256x256 for display)
            - Inference: DDIM sampling for fast CPU generation
            - Training: Conditional diffusion with sketch concatenation

            Built as part of the PyTorch Tutorial series.
            """
        )

    return demo


def main():
    """Launch the Gradio demo."""
    print("\n" + "=" * 50)
    print("Starting Sketch-to-Image Demo")
    print("=" * 50)

    demo = build_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
