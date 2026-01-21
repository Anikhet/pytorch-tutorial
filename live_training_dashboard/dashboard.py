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
from typing import Dict, Any, Optional

from neural_network import MonitoredMLP, MonitoredCNN

from training_monitor import (
    TrainingMonitor, TrainingHistory,
    create_spiral_loaders, create_mnist_loaders
)

# Constants for dead neuron detection
DEAD_NEURON_THRESHOLD = 1e-6
DEAD_NEURON_PERCENTAGE_THRESHOLD = 10.0
MIN_WEIGHT_STD_THRESHOLD = 1e-5
MAX_ANIMATION_FRAMES = 20
ANIMATION_DURATION_MS = 300


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
            window = max(1, min(50, len(history.batch_losses) // 5))
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
            window = max(1, min(50, len(history.grad_norms) // 5))
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


def detect_dead_neurons(history: TrainingHistory) -> Dict[str, Dict[str, Any]]:
    """
    Detect dead neurons by analyzing weight distributions.

    A neuron is considered "dead" if its weights are stuck near zero
    (absolute mean < threshold) across multiple snapshots.

    Returns:
        Dictionary mapping layer names to dead neuron statistics.
    """
    if not history.snapshots or len(history.snapshots) < 2:
        return {}

    dead_neuron_info: Dict[str, Dict[str, Any]] = {}

    latest = history.snapshots[-1]
    if latest.model_stats is None:
        return {}

    for layer_name, weights in latest.model_stats.weight_histograms.items():
        near_zero_count = np.sum(np.abs(weights) < DEAD_NEURON_THRESHOLD)
        total_weights = len(weights)
        dead_percentage = (near_zero_count / total_weights) * 100 if total_weights > 0 else 0

        weight_std = float(np.std(weights))
        weight_mean = float(np.mean(np.abs(weights)))

        is_problematic = (
            dead_percentage > DEAD_NEURON_PERCENTAGE_THRESHOLD or
            weight_std < MIN_WEIGHT_STD_THRESHOLD
        )

        dead_neuron_info[layer_name] = {
            'dead_percentage': dead_percentage,
            'near_zero_count': int(near_zero_count),
            'total_weights': total_weights,
            'weight_std': weight_std,
            'weight_mean': weight_mean,
            'is_problematic': is_problematic
        }

    return dead_neuron_info


def create_weight_evolution_chart(
    history: TrainingHistory,
    selected_layer: Optional[str] = None
) -> go.Figure:
    """
    Create animated weight distribution evolution chart.

    Shows how weight distributions shift from initialization through
    convergence using animated histogram frames.

    Args:
        history: Training history with snapshots
        selected_layer: Specific layer to visualize (or None for first layer)

    Returns:
        Plotly figure with animation controls
    """
    snapshots_with_stats = [
        s for s in history.snapshots
        if s.model_stats is not None and s.model_stats.weight_histograms
    ]

    if not snapshots_with_stats:
        fig = go.Figure()
        fig.add_annotation(
            text="No weight data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    layer_names = list(snapshots_with_stats[0].model_stats.weight_histograms.keys())
    if not layer_names:
        return go.Figure()

    if selected_layer is None or selected_layer not in layer_names:
        selected_layer = layer_names[0]

    sample_step = max(1, len(snapshots_with_stats) // MAX_ANIMATION_FRAMES)
    sampled_snapshots = snapshots_with_stats[::sample_step]
    if snapshots_with_stats[-1] not in sampled_snapshots:
        sampled_snapshots.append(snapshots_with_stats[-1])

    all_weights = []
    for snapshot in sampled_snapshots:
        weights = snapshot.model_stats.weight_histograms.get(selected_layer)
        if weights is not None:
            all_weights.extend(weights)

    if not all_weights:
        return go.Figure()

    weight_min = np.percentile(all_weights, 1)
    weight_max = np.percentile(all_weights, 99)
    bin_edges = np.linspace(weight_min, weight_max, 51)

    frames = []
    for idx, snapshot in enumerate(sampled_snapshots):
        weights = snapshot.model_stats.weight_histograms.get(selected_layer)
        if weights is None:
            continue

        counts, _ = np.histogram(weights, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        frame = go.Frame(
            data=[go.Bar(
                x=bin_centers,
                y=counts,
                marker_color='steelblue',
                marker_line_color='darkblue',
                marker_line_width=0.5,
                name=selected_layer
            )],
            name=f"epoch_{snapshot.epoch}_batch_{snapshot.batch}",
            layout=go.Layout(
                title=dict(
                    text=f"Weight Distribution - {selected_layer}<br>"
                         f"<sub>Epoch {snapshot.epoch}, Batch {snapshot.batch}</sub>"
                )
            )
        )
        frames.append(frame)

    first_weights = sampled_snapshots[0].model_stats.weight_histograms.get(selected_layer)
    first_counts, _ = np.histogram(first_weights, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure(
        data=[go.Bar(
            x=bin_centers,
            y=first_counts,
            marker_color='steelblue',
            marker_line_color='darkblue',
            marker_line_width=0.5,
            name=selected_layer
        )],
        frames=frames
    )

    fig.update_layout(
        title=f"Weight Distribution Evolution - {selected_layer}",
        xaxis_title="Weight Value",
        yaxis_title="Count",
        height=400,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=ANIMATION_DURATION_MS, redraw=True),
                                fromcurrent=True,
                                mode="immediate"
                            )
                        ]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate"
                            )
                        ]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12),
                    prefix="Training Progress: ",
                    visible=True,
                    xanchor="center"
                ),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.05,
                y=0,
                steps=[
                    dict(
                        args=[
                            [f"epoch_{s.epoch}_batch_{s.batch}"],
                            dict(
                                frame=dict(duration=ANIMATION_DURATION_MS, redraw=True),
                                mode="immediate"
                            )
                        ],
                        label=f"E{s.epoch}B{s.batch}",
                        method="animate"
                    )
                    for s in sampled_snapshots
                    if s.model_stats.weight_histograms.get(selected_layer) is not None
                ]
            )
        ]
    )

    return fig


def create_weight_evolution_comparison(history: TrainingHistory) -> go.Figure:
    """
    Create side-by-side comparison of initial vs final weight distributions.

    Shows how distributions shifted from initialization to convergence.
    """
    snapshots_with_stats = [
        s for s in history.snapshots
        if s.model_stats is not None and s.model_stats.weight_histograms
    ]

    if len(snapshots_with_stats) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 snapshots for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    initial_snapshot = snapshots_with_stats[0]
    final_snapshot = snapshots_with_stats[-1]

    layer_names = list(initial_snapshot.model_stats.weight_histograms.keys())
    n_layers = len(layer_names)

    if n_layers == 0:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=n_layers,
        subplot_titles=[f"{name} (Init)" for name in layer_names] +
                       [f"{name} (Final)" for name in layer_names],
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )

    for i, layer_name in enumerate(layer_names):
        init_weights = initial_snapshot.model_stats.weight_histograms.get(layer_name, [])
        final_weights = final_snapshot.model_stats.weight_histograms.get(layer_name, [])

        fig.add_trace(
            go.Histogram(
                x=init_weights, nbinsx=40,
                marker_color='lightcoral',
                name=f"{layer_name} Init",
                showlegend=False
            ),
            row=1, col=i+1
        )

        fig.add_trace(
            go.Histogram(
                x=final_weights, nbinsx=40,
                marker_color='steelblue',
                name=f"{layer_name} Final",
                showlegend=False
            ),
            row=2, col=i+1
        )

    fig.update_layout(
        title=f"Weight Evolution: Epoch {initial_snapshot.epoch} → Epoch {final_snapshot.epoch}",
        height=500,
        showlegend=False
    )

    return fig


def create_weight_statistics_evolution(history: TrainingHistory) -> go.Figure:
    """
    Create line chart showing weight statistics (mean, std) evolution over time.
    """
    snapshots_with_stats = [
        s for s in history.snapshots
        if s.model_stats is not None and s.model_stats.layer_stats
    ]

    if not snapshots_with_stats:
        return go.Figure()

    layer_names = list(snapshots_with_stats[0].model_stats.layer_stats.keys())
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Weight Mean (absolute)", "Weight Std Dev"]
    )

    for idx, layer_name in enumerate(layer_names):
        epochs = []
        weight_means = []
        weight_stds = []

        for snapshot in snapshots_with_stats:
            stats = snapshot.model_stats.layer_stats.get(layer_name)
            if stats:
                epochs.append(snapshot.epoch + snapshot.batch / 100)
                weight_means.append(abs(stats.weight_mean))
                weight_stds.append(stats.weight_std)

        color = colors[idx % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=epochs, y=weight_means,
                mode='lines', name=layer_name,
                line=dict(color=color),
                showlegend=True
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=epochs, y=weight_stds,
                mode='lines', name=layer_name,
                line=dict(color=color),
                showlegend=False
            ),
            row=1, col=2
        )

    fig.update_layout(
        title="Weight Statistics Over Training",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_xaxes(title_text="Training Progress (Epoch)", row=1, col=1)
    fig.update_xaxes(title_text="Training Progress (Epoch)", row=1, col=2)

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

    # Histograms and Weight Evolution
    st.subheader("📊 Distribution Analysis")
    tab1, tab2, tab3 = st.tabs([
        "Gradient Distributions",
        "Weight Distributions",
        "Weight Evolution"
    ])

    with tab1:
        st.plotly_chart(create_gradient_histogram(history), use_container_width=True)

    with tab2:
        st.plotly_chart(create_weight_histogram(history), use_container_width=True)

        # Dead neuron detection
        dead_neuron_info = detect_dead_neurons(history)
        if dead_neuron_info:
            has_issues = any(info['is_problematic'] for info in dead_neuron_info.values())
            if has_issues:
                st.warning("Dead Neuron Warning: Some layers show signs of dead neurons")

            with st.expander("Dead Neuron Analysis", expanded=has_issues):
                cols = st.columns(len(dead_neuron_info))
                for i, (layer_name, info) in enumerate(dead_neuron_info.items()):
                    with cols[i]:
                        status_icon = "⚠️" if info['is_problematic'] else "✓"
                        st.markdown(f"**{status_icon} {layer_name}**")
                        st.caption(f"Near-zero: {info['dead_percentage']:.2f}%")
                        st.caption(f"Std Dev: {info['weight_std']:.6f}")
                        if info['is_problematic']:
                            st.error("Potential dead neurons detected")

    with tab3:
        st.markdown("*Animated visualization of how weight distributions evolve during training*")

        # Layer selector for animation
        if history.snapshots:
            latest = history.snapshots[-1]
            if latest.model_stats and latest.model_stats.weight_histograms:
                layer_names = list(latest.model_stats.weight_histograms.keys())
                selected_layer = st.selectbox(
                    "Select Layer",
                    options=layer_names,
                    key="weight_evolution_layer"
                )

                # Animated weight evolution chart
                st.plotly_chart(
                    create_weight_evolution_chart(history, selected_layer),
                    use_container_width=True
                )

                # Weight statistics evolution
                st.plotly_chart(
                    create_weight_statistics_evolution(history),
                    use_container_width=True
                )

                # Initial vs Final comparison
                st.subheader("Initialization vs Convergence")
                st.plotly_chart(
                    create_weight_evolution_comparison(history),
                    use_container_width=True
                )
            else:
                st.info("Train the model to see weight evolution visualizations")
        else:
            st.info("Train the model to see weight evolution visualizations")

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
        - Use Weight Evolution tab to watch distributions shift from initialization to convergence
        - Dead neuron warnings appear when weights are stuck near zero (>10% or std < 1e-5)
        - Play the animation to see how weight distributions change through training epochs
        """
    )


if __name__ == "__main__":
    main()
