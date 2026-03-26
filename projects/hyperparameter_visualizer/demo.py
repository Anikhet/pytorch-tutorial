"""
Interactive Hyperparameter Tuning Visualizer.

Gradio interface for experimenting with:
- Learning rate
- Batch size
- Optimizer choice
- Network architecture
- Learning rate schedulers

Watch training progress in real-time!
"""

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import torch
from io import BytesIO
from PIL import Image
import threading

from neural_network import SimpleMLP, DeepMLP, DATASETS
from trainer import run_training, TrainingMetrics


# Global state for training
current_metrics = None
training_thread = None
stop_flag = False


def create_loss_plot(metrics: TrainingMetrics) -> Image.Image:
    """Create loss curve plot."""
    fig, ax = plt.subplots(figsize=(8, 4))

    if len(metrics.train_losses) > 0:
        epochs = range(1, len(metrics.train_losses) + 1)
        ax.plot(epochs, metrics.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, metrics.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add current values annotation
        if len(metrics.train_losses) > 0:
            ax.annotate(
                f'Train: {metrics.train_losses[-1]:.4f}\nVal: {metrics.val_losses[-1]:.4f}',
                xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
    else:
        ax.text(0.5, 0.5, 'Waiting for training...', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def create_accuracy_plot(metrics: TrainingMetrics) -> Image.Image:
    """Create accuracy curve plot."""
    fig, ax = plt.subplots(figsize=(8, 4))

    if len(metrics.train_accuracies) > 0:
        epochs = range(1, len(metrics.train_accuracies) + 1)
        ax.plot(epochs, [a * 100 for a in metrics.train_accuracies],
                'b-', label='Train Acc', linewidth=2)
        ax.plot(epochs, [a * 100 for a in metrics.val_accuracies],
                'r-', label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy Progress', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        # Add current values
        if len(metrics.train_accuracies) > 0:
            ax.annotate(
                f'Train: {metrics.train_accuracies[-1]*100:.1f}%\nVal: {metrics.val_accuracies[-1]*100:.1f}%',
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
            )
    else:
        ax.text(0.5, 0.5, 'Waiting for training...', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def create_lr_plot(metrics: TrainingMetrics) -> Image.Image:
    """Create learning rate schedule plot."""
    fig, ax = plt.subplots(figsize=(8, 4))

    if len(metrics.learning_rates) > 0:
        epochs = range(1, len(metrics.learning_rates) + 1)
        ax.plot(epochs, metrics.learning_rates, 'g-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        ax.annotate(
            f'Current LR: {metrics.learning_rates[-1]:.2e}',
            xy=(0.98, 0.98), xycoords='axes fraction',
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )
    else:
        ax.text(0.5, 0.5, 'Waiting for training...', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def create_data_plot(X: np.ndarray, y: np.ndarray) -> Image.Image:
    """Create dataset visualization plot."""
    fig, ax = plt.subplots(figsize=(6, 6))

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm',
                        alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Dataset Visualization', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def train_model(
    dataset_name: str,
    n_samples: int,
    noise: float,
    hidden_size: int,
    num_layers: int,
    optimizer_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    weight_decay: float,
    scheduler_name: str,
    activation: str,
    progress=gr.Progress()
):
    """Run training with given hyperparameters."""
    global current_metrics, stop_flag

    stop_flag = False

    # Generate dataset
    gen_fn = DATASETS.get(dataset_name, DATASETS["spiral"])
    X, y = gen_fn(n_samples=int(n_samples), noise=noise)

    # Split data
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Create model
    if num_layers <= 2:
        hidden_sizes = tuple([int(hidden_size)] * int(num_layers))
        model = SimpleMLP(
            input_size=2,
            hidden_sizes=hidden_sizes,
            output_size=2,
            activation=activation
        )
    else:
        model = DeepMLP(
            input_size=2,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            output_size=2,
            activation=activation
        )

    # Initialize metrics
    current_metrics = TrainingMetrics()
    current_metrics.total_epochs = int(num_epochs)

    # Progress callback
    def update_progress(metrics: TrainingMetrics):
        global stop_flag
        if stop_flag:
            metrics.should_stop = True
        progress(metrics.current_epoch / metrics.total_epochs,
                desc=f"Epoch {metrics.current_epoch}/{metrics.total_epochs}")

    # Run training
    metrics = run_training(
        X_train, y_train, X_val, y_val,
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        batch_size=int(batch_size),
        num_epochs=int(num_epochs),
        weight_decay=weight_decay,
        scheduler_name=scheduler_name,
        callback=update_progress
    )

    current_metrics = metrics

    # Create final plots
    loss_plot = create_loss_plot(metrics)
    acc_plot = create_accuracy_plot(metrics)
    lr_plot = create_lr_plot(metrics)
    data_plot = create_data_plot(X.numpy(), y.numpy())

    # Summary text
    final_train_acc = metrics.train_accuracies[-1] * 100 if metrics.train_accuracies else 0
    final_val_acc = metrics.val_accuracies[-1] * 100 if metrics.val_accuracies else 0
    final_train_loss = metrics.train_losses[-1] if metrics.train_losses else 0
    final_val_loss = metrics.val_losses[-1] if metrics.val_losses else 0
    total_time = sum(metrics.epoch_times)

    summary = f"""
## Training Complete!

### Final Results
| Metric | Train | Validation |
|--------|-------|------------|
| Loss | {final_train_loss:.4f} | {final_val_loss:.4f} |
| Accuracy | {final_train_acc:.1f}% | {final_val_acc:.1f}% |

### Configuration
- **Optimizer**: {optimizer_name}
- **Learning Rate**: {learning_rate}
- **Batch Size**: {batch_size}
- **Epochs**: {num_epochs}
- **Hidden Size**: {hidden_size}
- **Layers**: {num_layers}
- **Scheduler**: {scheduler_name}

### Training Time
- Total: {total_time:.1f}s
- Per Epoch: {total_time/num_epochs:.3f}s
"""

    return loss_plot, acc_plot, lr_plot, data_plot, summary


def stop_training():
    """Stop current training."""
    global stop_flag
    stop_flag = True
    return "Training stop requested..."


def build_interface():
    """Build the Gradio interface."""

    with gr.Blocks(
        title="Hyperparameter Tuning Visualizer",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # Hyperparameter Tuning Visualizer

            **Experiment with different hyperparameters and watch training in real-time!**

            Adjust the sliders below and click "Start Training" to see how different
            configurations affect model convergence and accuracy.
            """
        )

        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                gr.Markdown("### Dataset Settings")
                dataset_choice = gr.Dropdown(
                    choices=["spiral", "circles", "moons", "xor"],
                    value="spiral",
                    label="Dataset"
                )
                n_samples = gr.Slider(
                    minimum=200, maximum=2000, value=1000, step=100,
                    label="Number of Samples"
                )
                noise = gr.Slider(
                    minimum=0.0, maximum=0.5, value=0.2, step=0.05,
                    label="Noise Level"
                )

                gr.Markdown("### Model Architecture")
                hidden_size = gr.Slider(
                    minimum=8, maximum=128, value=32, step=8,
                    label="Hidden Layer Size"
                )
                num_layers = gr.Slider(
                    minimum=1, maximum=6, value=2, step=1,
                    label="Number of Hidden Layers"
                )
                activation = gr.Dropdown(
                    choices=["relu", "tanh", "leaky_relu", "gelu"],
                    value="relu",
                    label="Activation Function"
                )

                gr.Markdown("### Optimization")
                optimizer_choice = gr.Dropdown(
                    choices=["adam", "sgd", "adamw", "rmsprop"],
                    value="adam",
                    label="Optimizer"
                )
                learning_rate = gr.Slider(
                    minimum=0.0001, maximum=0.5, value=0.01, step=0.0001,
                    label="Learning Rate"
                )
                batch_size = gr.Slider(
                    minimum=8, maximum=256, value=32, step=8,
                    label="Batch Size"
                )
                num_epochs = gr.Slider(
                    minimum=10, maximum=200, value=50, step=10,
                    label="Number of Epochs"
                )

                gr.Markdown("### Regularization")
                weight_decay = gr.Slider(
                    minimum=0.0, maximum=0.1, value=0.0, step=0.001,
                    label="Weight Decay (L2)"
                )
                scheduler_choice = gr.Dropdown(
                    choices=["none", "step", "cosine", "exponential"],
                    value="none",
                    label="LR Scheduler"
                )

                with gr.Row():
                    train_btn = gr.Button("Start Training", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop", variant="stop", size="lg")

            # Right column - Visualizations
            with gr.Column(scale=2):
                with gr.Row():
                    loss_plot = gr.Image(label="Loss Curves", height=300)
                    acc_plot = gr.Image(label="Accuracy Curves", height=300)

                with gr.Row():
                    lr_plot = gr.Image(label="Learning Rate", height=300)
                    data_plot = gr.Image(label="Dataset", height=300)

                summary_output = gr.Markdown(label="Training Summary")

        # Tips section
        gr.Markdown(
            """
            ---
            ### Tips for Hyperparameter Tuning

            **Learning Rate:**
            - Too high: Loss oscillates or explodes
            - Too low: Training is slow, may get stuck
            - Sweet spot: Usually 0.001 - 0.01 for Adam

            **Batch Size:**
            - Small (8-32): More noise, better generalization
            - Large (128-256): Faster, but may overfit

            **Optimizer:**
            - Adam: Good default, adaptive learning rates
            - SGD: Simpler, may generalize better with momentum
            - AdamW: Adam with decoupled weight decay

            **Try these experiments:**
            1. High LR (0.1) vs Low LR (0.001) with SGD
            2. Adam vs SGD on the spiral dataset
            3. Cosine scheduler vs constant LR
            4. Deep network (5 layers) vs shallow (2 layers)
            """
        )

        # Event handlers
        train_btn.click(
            fn=train_model,
            inputs=[
                dataset_choice, n_samples, noise,
                hidden_size, num_layers,
                optimizer_choice, learning_rate, batch_size,
                num_epochs, weight_decay, scheduler_choice, activation
            ],
            outputs=[loss_plot, acc_plot, lr_plot, data_plot, summary_output]
        )

        stop_btn.click(
            fn=stop_training,
            outputs=summary_output
        )

    return demo


def main():
    """Launch the demo."""
    print("\n" + "=" * 50)
    print("Hyperparameter Tuning Visualizer")
    print("=" * 50)

    demo = build_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True
    )


if __name__ == "__main__":
    main()
