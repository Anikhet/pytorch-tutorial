"""
Interactive Neural Cellular Automata Visualization

A beautiful web interface to watch patterns grow, apply damage,
and observe self-repair in real-time.

Run with: python app.py
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple

from nca_model import NeuralCellularAutomata
from trainer import create_geometric_target, load_emoji, NCATrainer


class AppState:
    """Global state for the NCA application."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[NeuralCellularAutomata] = None
        self.state: Optional[torch.Tensor] = None
        self.target: Optional[torch.Tensor] = None
        self.size = 64
        self.is_running = False
        self.is_training = False

    def initialize_model(self, checkpoint_path: Optional[str] = None):
        """Initialize or load the NCA model."""
        self.model = NeuralCellularAutomata(
            state_channels=16,
            hidden_channels=128,
            cell_fire_rate=0.5,
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()

    def reset_state(self):
        """Reset to seed state."""
        if self.model is not None:
            self.state = self.model.create_seed(self.size, self.size, 1, self.device)

    def get_current_image(self) -> np.ndarray:
        """Get current state as RGB image."""
        if self.state is None:
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)

        with torch.no_grad():
            rgb = self.model.get_rgb(self.state)
            img = rgb[0].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def step(self, n_steps: int = 1) -> np.ndarray:
        """Run NCA for n steps and return image."""
        if self.model is None or self.state is None:
            return self.get_current_image()

        with torch.no_grad():
            self.state = self.model(self.state, steps=n_steps)
        return self.get_current_image()

    def apply_damage(self, damage_type: str, x: int = None, y: int = None) -> np.ndarray:
        """Apply damage and return image."""
        if self.state is None:
            return self.get_current_image()

        with torch.no_grad():
            if damage_type == "click" and x is not None and y is not None:
                self.state = self.model.damage(
                    self.state, "circle",
                    cx=x, cy=y, radius=self.size // 6
                )
            else:
                self.state = self.model.damage(self.state, damage_type)

        return self.get_current_image()


app_state = AppState()


def get_target_image(target_type: str, size: int = 64) -> np.ndarray:
    """Get target image for display."""
    if target_type in ["circle", "square", "triangle", "star", "heart"]:
        target = create_geometric_target(target_type, size)
    else:
        target = create_geometric_target("circle", size)

    rgba = target[0].permute(1, 2, 0).numpy()
    rgb = rgba[:, :, :3] * rgba[:, :, 3:4]
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def train_model(target_type: str, n_steps: int, progress=gr.Progress()) -> Tuple[str, np.ndarray]:
    """Train model on selected target."""
    app_state.is_training = True

    if target_type in ["circle", "square", "triangle", "star", "heart"]:
        target = create_geometric_target(target_type, app_state.size)
    else:
        target = create_geometric_target("circle", app_state.size)

    app_state.target = target.to(app_state.device)
    app_state.initialize_model()

    trainer = NCATrainer(
        model=app_state.model,
        target=target,
        device=app_state.device,
        lr=2e-3,
        pool_size=256,
        batch_size=8,
    )

    losses = []
    for step in progress.tqdm(range(n_steps), desc="Training"):
        damage_prob = 0.3 if step >= n_steps // 2 else 0.0
        loss = trainer.train_step((64, 96), damage_prob)
        losses.append(loss)

        if step % 100 == 0:
            progress(step / n_steps, desc=f"Loss: {loss:.4f}")

    save_path = Path("checkpoints") / f"nca_{target_type}.pth"
    save_path.parent.mkdir(exist_ok=True)
    trainer.save_checkpoint(str(save_path))

    app_state.reset_state()
    app_state.model.eval()

    for _ in range(100):
        app_state.step(1)

    app_state.is_training = False

    final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    return f"Training complete! Final loss: {final_loss:.4f}", app_state.get_current_image()


def load_model(target_type: str) -> Tuple[str, np.ndarray]:
    """Load pre-trained model if available."""
    checkpoint_path = Path("checkpoints") / f"nca_{target_type}.pth"

    if checkpoint_path.exists():
        app_state.initialize_model(str(checkpoint_path))
        app_state.reset_state()

        for _ in range(100):
            app_state.step(1)

        return f"Loaded model for {target_type}", app_state.get_current_image()
    else:
        return f"No checkpoint found for {target_type}. Please train first.", get_target_image(target_type)


def reset_simulation() -> np.ndarray:
    """Reset to seed."""
    app_state.reset_state()
    return app_state.get_current_image()


def run_steps(n_steps: int) -> np.ndarray:
    """Run N steps."""
    return app_state.step(n_steps)


def apply_damage_type(damage_type: str) -> np.ndarray:
    """Apply damage."""
    return app_state.apply_damage(damage_type)


def create_growth_video(progress=gr.Progress()) -> str:
    """Create a video of the growth process."""
    if app_state.model is None:
        return None

    frames = []
    app_state.reset_state()

    for i in progress.tqdm(range(200), desc="Generating frames"):
        app_state.step(1)
        if i % 2 == 0:
            img = app_state.get_current_image()
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((256, 256), Image.NEAREST)
            frames.append(np.array(pil_img))

    output_path = "growth_animation.gif"
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=50,
        loop=0
    )

    return output_path


def create_regeneration_video(damage_type: str = "half", progress=gr.Progress()) -> str:
    """Create a video showing regeneration after damage."""
    if app_state.model is None:
        return None

    frames = []

    app_state.reset_state()
    for _ in range(100):
        app_state.step(1)

    for _ in range(20):
        img = app_state.get_current_image()
        frames.append(Image.fromarray(img).resize((256, 256), Image.NEAREST))

    app_state.apply_damage(damage_type)

    for _ in range(10):
        img = app_state.get_current_image()
        frames.append(Image.fromarray(img).resize((256, 256), Image.NEAREST))

    for i in progress.tqdm(range(150), desc="Regenerating"):
        app_state.step(1)
        img = app_state.get_current_image()
        frames.append(Image.fromarray(img).resize((256, 256), Image.NEAREST))

    for _ in range(20):
        frames.append(frames[-1])

    output_path = "regeneration_animation.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0
    )

    return output_path


def create_interface():
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Neural Cellular Automata",
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="blue"),
    ) as demo:
        gr.Markdown("""
        # Neural Cellular Automata

        Watch self-organizing patterns grow from a single cell!
        These digital creatures can **regenerate** when damaged.

        Based on [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
        """)

        with gr.Tabs():
            with gr.Tab("Playground"):
                with gr.Row():
                    with gr.Column(scale=2):
                        main_image = gr.Image(
                            label="NCA State",
                            value=np.zeros((256, 256, 3), dtype=np.uint8),
                            height=400,
                            width=400,
                            interactive=False,
                        )

                    with gr.Column(scale=1):
                        target_select = gr.Dropdown(
                            choices=["heart", "star", "circle", "square", "triangle"],
                            value="heart",
                            label="Target Shape",
                        )
                        target_preview = gr.Image(
                            label="Target",
                            value=get_target_image("heart"),
                            height=100,
                            width=100,
                        )

                        gr.Markdown("### Controls")

                        with gr.Row():
                            reset_btn = gr.Button("Reset (Seed)", variant="primary")
                            step_btn = gr.Button("Step x10")

                        with gr.Row():
                            grow_btn = gr.Button("Grow x100", variant="secondary")

                        gr.Markdown("### Apply Damage")
                        with gr.Row():
                            damage_half = gr.Button("Half")
                            damage_circle = gr.Button("Circle")
                            damage_random = gr.Button("Random")

                        status_text = gr.Textbox(
                            label="Status",
                            value="Select a target and train the model",
                            interactive=False
                        )

                target_select.change(
                    fn=lambda t: get_target_image(t),
                    inputs=[target_select],
                    outputs=[target_preview]
                )

                reset_btn.click(reset_simulation, outputs=[main_image])
                step_btn.click(lambda: run_steps(10), outputs=[main_image])
                grow_btn.click(lambda: run_steps(100), outputs=[main_image])

                damage_half.click(lambda: apply_damage_type("half"), outputs=[main_image])
                damage_circle.click(lambda: apply_damage_type("circle"), outputs=[main_image])
                damage_random.click(lambda: apply_damage_type("random"), outputs=[main_image])

            with gr.Tab("Training"):
                gr.Markdown("""
                ### Train Your Own NCA

                Select a target shape and train the model to grow it from a seed.
                """)

                with gr.Row():
                    train_target = gr.Dropdown(
                        choices=["heart", "star", "circle", "square", "triangle"],
                        value="heart",
                        label="Target Shape",
                    )
                    train_steps = gr.Slider(
                        minimum=500,
                        maximum=5000,
                        value=2000,
                        step=500,
                        label="Training Steps",
                    )

                with gr.Row():
                    train_btn = gr.Button("Start Training", variant="primary")
                    load_btn = gr.Button("Load Existing", variant="secondary")

                train_output = gr.Image(label="Training Result", height=256)
                train_status = gr.Textbox(label="Training Status")

                train_btn.click(
                    train_model,
                    inputs=[train_target, train_steps],
                    outputs=[train_status, train_output]
                )

                load_btn.click(
                    load_model,
                    inputs=[train_target],
                    outputs=[train_status, train_output]
                )

            with gr.Tab("Animations"):
                gr.Markdown("""
                ### Generate Animations

                Create mesmerizing animations of growth and regeneration.
                """)

                with gr.Row():
                    growth_video_btn = gr.Button("Generate Growth Animation", variant="primary")
                    regen_video_btn = gr.Button("Generate Regeneration Animation", variant="secondary")

                regen_damage_type = gr.Dropdown(
                    choices=["half", "circle", "random"],
                    value="half",
                    label="Damage Type for Regeneration",
                )

                animation_output = gr.Image(label="Animation (GIF)")

                growth_video_btn.click(create_growth_video, outputs=[animation_output])
                regen_video_btn.click(
                    create_regeneration_video,
                    inputs=[regen_damage_type],
                    outputs=[animation_output]
                )

            with gr.Tab("About"):
                gr.Markdown("""
                ## What is Neural Cellular Automata?

                Neural Cellular Automata (NCA) combines:
                - **Cellular Automata**: Simple local rules creating complex patterns
                - **Neural Networks**: Learned rules instead of hand-crafted ones

                ### How it works:

                1. **State**: Each pixel has a state vector (RGBA + hidden channels)
                2. **Perception**: Cells perceive neighbors using Sobel filters
                3. **Update**: A small neural network computes state changes
                4. **Stochastic**: Only some cells update each step

                ### Key Properties:

                - **Self-Organizing**: Patterns emerge from local rules
                - **Regeneration**: Damaged patterns can regrow
                - **Persistence**: Stable patterns maintain themselves

                ### References:

                - [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
                - [Self-Organising Textures](https://distill.pub/selforg/2021/textures/)

                Built with PyTorch and Gradio
                """)

    return demo


if __name__ == "__main__":
    app_state.initialize_model()
    app_state.reset_state()

    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
