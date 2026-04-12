"""
Neural Cellular Automata Demo

Run this script to train and visualize NCA in action.
Creates beautiful GIF animations of growth and regeneration.

Usage:
    python demo.py --target heart --steps 2000
    python demo.py --quick  # Fast demo with simple shape
"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from nca_model import NeuralCellularAutomata
from trainer import create_geometric_target, NCATrainer


def create_beautiful_animation(
    model: NeuralCellularAutomata,
    size: int = 64,
    n_growth_frames: int = 150,
    n_damage_frames: int = 100,
    output_path: str = "nca_demo.gif",
    device: str = "cuda",
):
    """Create a stunning animation showing growth and regeneration."""
    print("Generating animation frames...")

    frames = []
    model.train(False)

    state = model.create_seed(size, size, 1, device)

    with torch.no_grad():
        for i in tqdm(range(n_growth_frames), desc="Growing"):
            state = model(state, steps=1)
            rgb = model.get_rgb(state)
            img = rgb[0].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            frames.append(img)

    for _ in range(20):
        frames.append(frames[-1])

    with torch.no_grad():
        state = model.damage(state, "half", direction="right")
        rgb = model.get_rgb(state)
        img = rgb[0].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    for _ in range(15):
        frames.append(img)

    with torch.no_grad():
        for i in tqdm(range(n_damage_frames), desc="Regenerating"):
            state = model(state, steps=1)
            rgb = model.get_rgb(state)
            img = rgb[0].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            frames.append(img)

    for _ in range(30):
        frames.append(frames[-1])

    with torch.no_grad():
        state = model.damage(state, "circle", cx=size//2, cy=size//2, radius=size//4)
        rgb = model.get_rgb(state)
        img = rgb[0].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    for _ in range(15):
        frames.append(img)

    with torch.no_grad():
        for i in tqdm(range(n_damage_frames), desc="Regenerating again"):
            state = model(state, steps=1)
            rgb = model.get_rgb(state)
            img = rgb[0].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            frames.append(img)

    for _ in range(40):
        frames.append(frames[-1])

    print(f"Saving animation to {output_path}...")
    upscaled_frames = []
    for frame in frames:
        pil_img = Image.fromarray(frame)
        pil_img = pil_img.resize((256, 256), Image.NEAREST)
        upscaled_frames.append(pil_img)

    upscaled_frames[0].save(
        output_path,
        save_all=True,
        append_images=upscaled_frames[1:],
        duration=50,
        loop=0,
    )

    print(f"Animation saved to {output_path}")
    return output_path


def create_growth_grid(
    model: NeuralCellularAutomata,
    size: int = 64,
    output_path: str = "nca_growth_grid.png",
    device: str = "cuda",
):
    """Create a grid showing growth stages."""
    stages = [0, 20, 40, 60, 80, 100, 120, 150]

    model.train(False)
    state = model.create_seed(size, size, 1, device)
    images = []

    with torch.no_grad():
        for step in range(max(stages) + 1):
            state = model(state, steps=1)
            if step in stages:
                rgb = model.get_rgb(state)
                img = rgb[0].permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                images.append(img)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')

    for idx, (ax, img, step) in enumerate(zip(axes.flatten(), images, stages)):
        ax.imshow(img, interpolation='nearest')
        ax.set_title(f"Step {step}", color='white', fontsize=12)
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

    plt.suptitle("Neural Cellular Automata: Growth Stages", color='white', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()

    print(f"Growth grid saved to {output_path}")
    return output_path


def create_regeneration_grid(
    model: NeuralCellularAutomata,
    size: int = 64,
    output_path: str = "nca_regeneration_grid.png",
    device: str = "cuda",
):
    """Create a grid showing regeneration after damage."""
    model.train(False)

    state = model.create_seed(size, size, 1, device)
    with torch.no_grad():
        state = model(state, steps=100)

    rgb = model.get_rgb(state)
    grown_img = rgb[0].permute(1, 2, 0).cpu().numpy()
    grown_img = np.clip(grown_img, 0, 1)

    damaged_state = model.damage(state.clone(), "half", direction="right")
    rgb = model.get_rgb(damaged_state)
    damaged_img = rgb[0].permute(1, 2, 0).cpu().numpy()
    damaged_img = np.clip(damaged_img, 0, 1)

    regen_stages = [20, 40, 60, 100]
    regen_images = [damaged_img]

    current_state = damaged_state
    with torch.no_grad():
        for step in range(max(regen_stages) + 1):
            current_state = model(current_state, steps=1)
            if step in regen_stages:
                rgb = model.get_rgb(current_state)
                img = rgb[0].permute(1, 2, 0).cpu().numpy()
                regen_images.append(np.clip(img, 0, 1))

    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    fig.patch.set_facecolor('#1a1a2e')

    titles = ["Fully Grown", "Damaged", "Regen +20", "Regen +40", "Regen +60", "Regen +100"]
    all_images = [grown_img] + regen_images

    for ax, img, title in zip(axes.flatten(), all_images, titles):
        ax.imshow(img, interpolation='nearest')
        ax.set_title(title, color='white', fontsize=10)
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

    plt.suptitle("Neural Cellular Automata: Self-Repair", color='white', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()

    print(f"Regeneration grid saved to {output_path}")
    return output_path


def train_and_demo(
    target_type: str = "heart",
    n_steps: int = 2000,
    size: int = 64,
    device: str = "cuda",
    output_dir: str = "outputs",
):
    """Train model and create all demo visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Neural Cellular Automata Demo")
    print(f"Target: {target_type}, Training steps: {n_steps}")
    print(f"Device: {device}")
    print(f"{'='*50}\n")

    target = create_geometric_target(target_type, size)

    model = NeuralCellularAutomata(
        state_channels=16,
        hidden_channels=128,
        cell_fire_rate=0.5,
    ).to(device)

    trainer = NCATrainer(
        model=model,
        target=target,
        device=device,
        lr=2e-3,
        pool_size=256,
        batch_size=8,
    )

    print("Training NCA...")
    trainer.train(
        n_steps=n_steps,
        steps_range=(64, 96),
        damage_start=n_steps // 2,
        damage_prob=0.3,
        log_every=200,
        save_every=0,
        save_dir=str(output_path / "checkpoints"),
    )

    print("\nGenerating visualizations...")

    create_beautiful_animation(
        model,
        size=size,
        output_path=str(output_path / f"nca_{target_type}_demo.gif"),
        device=device,
    )

    create_growth_grid(
        model,
        size=size,
        output_path=str(output_path / f"nca_{target_type}_growth.png"),
        device=device,
    )

    create_regeneration_grid(
        model,
        size=size,
        output_path=str(output_path / f"nca_{target_type}_regeneration.png"),
        device=device,
    )

    print(f"\nAll outputs saved to {output_path}/")
    print("Done!")


def quick_demo(device: str = "cuda"):
    """Quick demo with minimal training for testing."""
    print("Running quick demo (500 steps)...")
    train_and_demo(
        target_type="circle",
        n_steps=500,
        size=64,
        device=device,
        output_dir="outputs_quick",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Cellular Automata Demo")
    parser.add_argument(
        "--target",
        type=str,
        default="heart",
        choices=["circle", "square", "triangle", "star", "heart"],
        help="Target shape to grow",
    )
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--size", type=int, default=64, help="Image size")
    parser.add_argument("--quick", action="store_true", help="Run quick demo")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    if args.quick:
        quick_demo(args.device)
    else:
        train_and_demo(
            target_type=args.target,
            n_steps=args.steps,
            size=args.size,
            device=args.device,
            output_dir=args.output,
        )
