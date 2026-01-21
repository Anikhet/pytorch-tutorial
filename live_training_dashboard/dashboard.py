"""
Live Training Dashboard with Streamlit.

Visualizes:
- Real-time loss and accuracy curves
- Gradient histograms per layer
- Weight distributions
- Model architecture
- Training statistics
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
import time
import threading

from neural_network import MonitoredMLP, MonitoredCNN
from training_monitor import (
    TrainingMonitor, TrainingHistory,
    create_spiral_loaders, create_mnist_loaders
)


# Page configuration
st.set_page_config(
    page_title="Live Training Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_loss_chart(history: TrainingHistory) -> go.Figure:
    """Create interactive loss curve chart."""
    fig = go.Figure()

    if len(history.epoch_train_losses) > 0:
        epochs = list(range(1, len(history.epoch_train_losses) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history.epoch_train_losses,
            mode='lines+markers', name='Train Loss',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=history.epoch_val_losses,
            mode='lines+markers', name='Val Loss',
            line=dict(color='red', width=2)
        ))

    fig.update_layout(
        title='Loss Curves',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        height=350
    )
    return fig


def create_accuracy_chart(history: TrainingHistory) -> go.Figure:
    """Create interactive accuracy chart."""
    fig = go.Figure()

    if len(history.epoch_train_accs) > 0:
        epochs = list(range(1, len(history.epoch_train_accs) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=[a * 100 for a in history.epoch_train_accs],
            mode='lines+markers', name='Train Accuracy',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=[a * 100 for a in history.epoch_val_accs],
            mode='lines+markers', name='Val Accuracy',
            line=dict(color='red', width=2)
        ))

    fig.update_layout(
        title='Accuracy Curves',
        xaxis_title='Epoch',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0, 105]),
        hovermode='x unified',
        height=350
    )
    return fig


def create_batch_loss_chart(history: TrainingHistory) -> go.Figure:
    """Create batch-level loss chart."""
    fig = go.Figure()

    if len(history.batch_losses) > 0:
        fig.add_trace(go.Scatter(
            y=history.batch_losses,
            mode='lines', name='Batch Loss',
            line=dict(color='green', width=1),
            opacity=0.7
        ))

        # Add moving average
        if len(history.batch_losses) > 10:
            window = min(50, len(history.batch_losses) // 5)
            ma = pd.Series(history.batch_losses).rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                y=ma,
                mode='lines', name=f'Moving Avg ({window})',
                line=dict(color='darkgreen', width=2)
            ))

    fig.update_layout(
        title='Batch Loss (Training)',
        xaxis_title='Batch',
        yaxis_title='Loss',
        height=300
    )
    return fig


def create_gradient_norm_chart(history: TrainingHistory) -> go.Figure:
    """Create gradient norm chart."""
    fig = go.Figure()

    if len(history.grad_norms) > 0:
        fig.add_trace(go.Scatter(
            y=history.grad_norms,
            mode='lines', name='Gradient Norm',
            line=dict(color='purple', width=1),
            opacity=0.7
        ))

        # Add moving average
        if len(history.grad_norms) > 10:
            window = min(50, len(history.grad_norms) // 5)
            ma = pd.Series(history.grad_norms).rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                y=ma,
                mode='lines', name=f'Moving Avg ({window})',
                line=dict(color='darkviolet', width=2)
            ))

    fig.update_layout(
        title='Gradient Norm',
        xaxis_title='Batch',
        yaxis_title='Norm',
        height=300
    )
    return fig


def create_lr_chart(history: TrainingHistory) -> go.Figure:
    """Create learning rate schedule chart."""
    fig = go.Figure()

    if len(history.epoch_learning_rates) > 0:
        epochs = list(range(1, len(history.epoch_learning_rates) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history.epoch_learning_rates,
            mode='lines+markers', name='Learning Rate',
            line=dict(color='orange', width=2)
        ))

    fig.update_layout(
        title='Learning Rate Schedule',
        xaxis_title='Epoch',
        yaxis_title='Learning Rate',
        yaxis_type='log',
        height=300
    )
    return fig


def create_gradient_histogram(history: TrainingHistory) -> go.Figure:
    """Create gradient distribution histograms."""
    if not history.snapshots:
        return go.Figure()

    latest = history.snapshots[-1]
    if latest.model_stats is None:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=len(latest.model_stats.grad_histograms),
        subplot_titles=list(latest.model_stats.grad_histograms.keys())
    )

    for i, (name, grads) in enumerate(latest.model_stats.grad_histograms.items(), 1):
        fig.add_trace(
            go.Histogram(x=grads, name=name, nbinsx=50, showlegend=False),
            row=1, col=i
        )

    fig.update_layout(
        title='Gradient Distributions by Layer',
        height=300
    )
    return fig


def create_weight_histogram(history: TrainingHistory) -> go.Figure:
    """Create weight distribution histograms."""
    if not history.snapshots:
        return go.Figure()

    latest = history.snapshots[-1]
    if latest.model_stats is None:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=len(latest.model_stats.weight_histograms),
        subplot_titles=list(latest.model_stats.weight_histograms.keys())
    )

    colors = px.colors.qualitative.Plotly

    for i, (name, weights) in enumerate(latest.model_stats.weight_histograms.items(), 1):
        fig.add_trace(
            go.Histogram(
                x=weights, name=name, nbinsx=50,
                marker_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=1, col=i
        )

    fig.update_layout(
        title='Weight Distributions by Layer',
        height=300
    )
    return fig


def create_architecture_diagram(model) -> go.Figure:
    """Create a simple architecture visualization."""
    if hasattr(model, 'get_architecture_info'):
        layers = model.get_architecture_info()
    else:
        return go.Figure()

    # Create layer boxes
    fig = go.Figure()

    n_layers = len(layers)
    x_positions = np.linspace(0, 10, n_layers)
    max_size = max(layer.get('output_size', 10) for layer in layers)

    for i, layer in enumerate(layers):
        size = layer.get('output_size', 10)
        height = size / max_size * 2 + 0.5

        # Layer box
        fig.add_shape(
            type="rect",
            x0=x_positions[i] - 0.3, x1=x_positions[i] + 0.3,
            y0=-height/2, y1=height/2,
            fillcolor="lightblue" if layer['type'] == 'Linear' else "lightgreen",
            line=dict(color="darkblue", width=2)
        )

        # Layer label
        fig.add_annotation(
            x=x_positions[i], y=height/2 + 0.3,
            text=f"{layer['name']}<br>{layer['type']}<br>{layer['params']:,}p",
            showarrow=False,
            font=dict(size=10)
        )

        # Connection lines
        if i < n_layers - 1:
            fig.add_shape(
                type="line",
                x0=x_positions[i] + 0.3, x1=x_positions[i+1] - 0.3,
                y0=0, y1=0,
                line=dict(color="gray", width=1, dash="dot")
            )

    fig.update_layout(
        title='Model Architecture',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        height=250
    )
    return fig


def main():
    """Main dashboard application."""

    st.title("📊 Live Training Dashboard")
    st.markdown("*Real-time visualization of neural network training*")

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = TrainingHistory()
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'monitor' not in st.session_state:
        st.session_state.monitor = None

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Dataset")
        dataset = st.selectbox(
            "Dataset",
            ["Spiral (2D)", "MNIST-like"],
            help="Choose the training dataset"
        )

        st.subheader("Model Architecture")
        model_type = st.selectbox(
            "Model Type",
            ["MLP", "CNN (for MNIST-like)"],
            help="Choose model architecture"
        )

        if model_type == "MLP":
            hidden_layers = st.text_input(
                "Hidden Layers",
                value="128,64,32",
                help="Comma-separated layer sizes"
            )
            activation = st.selectbox(
                "Activation",
                ["relu", "tanh", "leaky_relu", "gelu"]
            )

        st.subheader("Training")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.01
        )
        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64, 128, 256],
            value=32
        )
        num_epochs = st.slider(
            "Epochs",
            min_value=5, max_value=100, value=30
        )
        optimizer_name = st.selectbox(
            "Optimizer",
            ["Adam", "SGD", "AdamW"]
        )
        scheduler_name = st.selectbox(
            "LR Scheduler",
            ["None", "StepLR", "CosineAnnealing"]
        )

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button(
                "▶️ Start",
                type="primary",
                disabled=st.session_state.is_training,
                use_container_width=True
            )
        with col2:
            stop_button = st.button(
                "⏹️ Stop",
                disabled=not st.session_state.is_training,
                use_container_width=True
            )

    # Handle start training
    if start_button and not st.session_state.is_training:
        # Parse hidden layers
        if model_type == "MLP":
            hidden_sizes = tuple(int(x.strip()) for x in hidden_layers.split(','))

            if dataset == "Spiral (2D)":
                input_size, output_size = 2, 3
            else:
                input_size, output_size = 784, 10

            model = MonitoredMLP(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=output_size,
                activation=activation
            )
        else:
            model = MonitoredCNN(input_channels=1, num_classes=10)

        # Create data loaders
        if dataset == "Spiral (2D)":
            train_loader, val_loader = create_spiral_loaders(batch_size=batch_size)
        else:
            train_loader, val_loader = create_mnist_loaders(batch_size=batch_size)

        # Create optimizer
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Create scheduler
        scheduler = None
        if scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        elif scheduler_name == "CosineAnnealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Create monitor
        monitor = TrainingMonitor(model, optimizer, scheduler, collect_every_n_batches=3)

        st.session_state.model = model
        st.session_state.monitor = monitor
        st.session_state.is_training = True

        # Run training
        with st.spinner("Training in progress..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_callback(history):
                progress = history.current_epoch / history.total_epochs
                progress_bar.progress(progress)
                status_text.text(
                    f"Epoch {history.current_epoch}/{history.total_epochs} | "
                    f"Train Loss: {history.epoch_train_losses[-1]:.4f} | "
                    f"Val Acc: {history.epoch_val_accs[-1]*100:.1f}%"
                )

            history = monitor.train(
                train_loader, val_loader,
                num_epochs=num_epochs,
                callback=update_callback
            )

            st.session_state.history = history
            st.session_state.is_training = False
            st.rerun()

    # Handle stop
    if stop_button and st.session_state.monitor:
        st.session_state.monitor.stop_training()
        st.session_state.is_training = False

    # Main dashboard area
    history = st.session_state.history

    # Top metrics
    if len(history.epoch_train_losses) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Current Epoch",
                f"{history.current_epoch}/{history.total_epochs}"
            )
        with col2:
            st.metric(
                "Train Loss",
                f"{history.epoch_train_losses[-1]:.4f}",
                delta=f"{history.epoch_train_losses[-1] - history.epoch_train_losses[-2]:.4f}" if len(history.epoch_train_losses) > 1 else None
            )
        with col3:
            st.metric(
                "Val Accuracy",
                f"{history.epoch_val_accs[-1]*100:.1f}%",
                delta=f"{(history.epoch_val_accs[-1] - history.epoch_val_accs[-2])*100:.1f}%" if len(history.epoch_val_accs) > 1 else None
            )
        with col4:
            if len(history.grad_norms) > 0:
                st.metric("Gradient Norm", f"{history.grad_norms[-1]:.4f}")

    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_loss_chart(history), use_container_width=True)
    with col2:
        st.plotly_chart(create_accuracy_chart(history), use_container_width=True)

    # Charts row 2
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_batch_loss_chart(history), use_container_width=True)
    with col2:
        st.plotly_chart(create_gradient_norm_chart(history), use_container_width=True)

    # Histograms
    st.subheader("📊 Distribution Analysis")
    tab1, tab2 = st.tabs(["Gradient Distributions", "Weight Distributions"])

    with tab1:
        st.plotly_chart(create_gradient_histogram(history), use_container_width=True)
    with tab2:
        st.plotly_chart(create_weight_histogram(history), use_container_width=True)

    # Architecture and stats
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🏗️ Model Architecture")
        if st.session_state.model is not None:
            st.plotly_chart(
                create_architecture_diagram(st.session_state.model),
                use_container_width=True
            )

            # Architecture details
            if hasattr(st.session_state.model, 'get_architecture_info'):
                arch_df = pd.DataFrame(st.session_state.model.get_architecture_info())
                st.dataframe(arch_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("📈 Layer Statistics")
        if st.session_state.monitor is not None:
            stats = st.session_state.monitor.get_gradient_stats_df()
            if stats:
                st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

    # Learning rate chart
    st.subheader("📉 Learning Rate Schedule")
    st.plotly_chart(create_lr_chart(history), use_container_width=True)

    # Footer
    st.divider()
    st.markdown(
        """
        **Tips:**
        - Watch for diverging train/val loss (overfitting)
        - Check gradient norms for exploding/vanishing gradients
        - Use histograms to spot dead neurons (all zeros)
        """
    )


if __name__ == "__main__":
    main()
