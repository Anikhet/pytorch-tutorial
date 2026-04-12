"""
3D Loss Landscape Visualizer Module

Creates interactive 3D visualizations of loss landscapes with optimizer
trajectory overlays. Features glowing paths, side-by-side comparisons,
and zoom controls for examining saddle points and local minima.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from landscape import LandscapeData, FUNCTION_REGISTRY
from optimizer_tracker import OptimizationPath


@dataclass
class VisualizerConfig:
    """Configuration for the visualizer."""
    surface_opacity: float = 0.85
    surface_colorscale: str = "Viridis"
    trajectory_width: float = 6.0
    marker_size: float = 8.0
    glow_layers: int = 3
    glow_intensity: float = 0.3
    show_start_marker: bool = True
    show_end_marker: bool = True
    show_gradient_arrows: bool = False
    z_scale: float = 1.0
    log_scale_z: bool = False


class LossLandscapeVisualizer:
    """
    Creates interactive 3D visualizations of loss landscapes.

    Features:
    - 3D surface plots with customizable colorscales
    - Glowing optimizer trajectories
    - Side-by-side comparison views
    - Interactive zoom and rotation
    - Markers for start points, end points, and known minima
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Initialize the visualizer.

        Args:
            config: Visualization configuration. Uses defaults if None.
        """
        self.config = config or VisualizerConfig()

    def create_surface(
        self,
        landscape: LandscapeData,
        opacity: Optional[float] = None,
    ) -> go.Surface:
        """
        Create a 3D surface plot of the loss landscape.

        Args:
            landscape: Computed landscape data
            opacity: Surface opacity (uses config default if None)

        Returns:
            Plotly Surface trace
        """
        z_values = landscape.z_values.copy()

        # Apply log scale if configured
        if self.config.log_scale_z:
            z_values = np.log1p(z_values - z_values.min() + 1)

        # Apply z-axis scaling
        z_values = z_values * self.config.z_scale

        return go.Surface(
            x=landscape.x_grid,
            y=landscape.y_grid,
            z=z_values,
            colorscale=self.config.surface_colorscale,
            opacity=opacity or self.config.surface_opacity,
            showscale=True,
            colorbar=dict(
                title="Loss",
                titleside="right",
                thickness=15,
                len=0.7,
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project_z=True,
                )
            ),
            hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>loss: %{z:.4f}<extra></extra>",
        )

    def create_trajectory(
        self,
        path: OptimizationPath,
        z_offset: float = 0.0,
    ) -> List[go.Scatter3d]:
        """
        Create a glowing 3D trajectory line with markers.

        The glow effect is created by layering multiple semi-transparent
        lines of increasing width behind the main trajectory.

        Args:
            path: Optimization path to visualize
            z_offset: Offset to lift trajectory above surface

        Returns:
            List of Plotly Scatter3d traces (glow layers + main line + markers)
        """
        traces = []
        x = path.x_coords
        y = path.y_coords
        z = path.z_coords

        # Apply transformations matching surface
        if self.config.log_scale_z:
            z_min = z.min()
            z = np.log1p(z - z_min + 1)

        z = z * self.config.z_scale + z_offset

        # Create glow layers (larger, more transparent lines behind main line)
        for i in range(self.config.glow_layers, 0, -1):
            glow_width = self.config.trajectory_width + (i * 4)
            glow_opacity = self.config.glow_intensity / i

            traces.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(
                        color=path.color,
                        width=glow_width,
                    ),
                    opacity=glow_opacity,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Main trajectory line
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(
                    color=path.color,
                    width=self.config.trajectory_width,
                ),
                name=path.optimizer_name,
                hovertemplate=(
                    f"<b>{path.optimizer_name}</b><br>"
                    "x: %{x:.3f}<br>"
                    "y: %{y:.3f}<br>"
                    "loss: %{z:.4f}<extra></extra>"
                ),
            )
        )

        # Start marker
        if self.config.show_start_marker and len(x) > 0:
            traces.append(
                go.Scatter3d(
                    x=[x[0]],
                    y=[y[0]],
                    z=[z[0]],
                    mode="markers",
                    marker=dict(
                        size=self.config.marker_size + 2,
                        color="white",
                        symbol="circle",
                        line=dict(color=path.color, width=3),
                    ),
                    name=f"{path.optimizer_name} Start",
                    showlegend=False,
                    hovertemplate=f"<b>{path.optimizer_name} Start</b><br>({x[0]:.3f}, {y[0]:.3f})<extra></extra>",
                )
            )

        # End marker
        if self.config.show_end_marker and len(x) > 0:
            traces.append(
                go.Scatter3d(
                    x=[x[-1]],
                    y=[y[-1]],
                    z=[z[-1]],
                    mode="markers",
                    marker=dict(
                        size=self.config.marker_size,
                        color=path.color,
                        symbol="diamond",
                    ),
                    name=f"{path.optimizer_name} End",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{path.optimizer_name} End</b><br>"
                        f"({x[-1]:.3f}, {y[-1]:.3f})<br>"
                        f"Loss: {path.final_loss:.6f}<extra></extra>"
                    ),
                )
            )

        return traces

    def create_minima_markers(
        self,
        function_name: str,
        landscape: LandscapeData,
    ) -> List[go.Scatter3d]:
        """
        Create markers for known global/local minima.

        Args:
            function_name: Name of the loss function
            landscape: Landscape data for z-value lookup

        Returns:
            List of Scatter3d traces for minima markers
        """
        if function_name not in FUNCTION_REGISTRY:
            return []

        func_info = FUNCTION_REGISTRY[function_name]
        minima = func_info.get("minima", [])
        saddle_points = func_info.get("saddle_points", [])

        traces = []

        # Add minima markers
        for i, (mx, my) in enumerate(minima):
            # Compute z value at minimum
            func = func_info["func"]
            mz = float(func(np.array([mx]), np.array([my]))[0])

            if self.config.log_scale_z:
                mz = np.log1p(mz - landscape.z_values.min() + 1)
            mz = mz * self.config.z_scale

            traces.append(
                go.Scatter3d(
                    x=[mx],
                    y=[my],
                    z=[mz],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color="gold",
                        symbol="diamond",
                        line=dict(color="black", width=2),
                    ),
                    name=f"Minimum {i+1}" if len(minima) > 1 else "Global Minimum",
                    hovertemplate=f"<b>Minimum</b><br>({mx:.3f}, {my:.3f})<br>Loss: {mz:.6f}<extra></extra>",
                )
            )

        # Add saddle point markers
        for i, (sx, sy) in enumerate(saddle_points):
            func = func_info["func"]
            sz = float(func(np.array([sx]), np.array([sy]))[0])

            if self.config.log_scale_z:
                sz = np.log1p(sz - landscape.z_values.min() + 1)
            sz = sz * self.config.z_scale

            traces.append(
                go.Scatter3d(
                    x=[sx],
                    y=[sy],
                    z=[sz],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="red",
                        symbol="x",
                        line=dict(color="white", width=1),
                    ),
                    name=f"Saddle Point {i+1}" if len(saddle_points) > 1 else "Saddle Point",
                    hovertemplate=f"<b>Saddle Point</b><br>({sx:.3f}, {sy:.3f})<extra></extra>",
                )
            )

        return traces

    def create_single_view(
        self,
        landscape: LandscapeData,
        paths: Optional[List[OptimizationPath]] = None,
        title: str = "Loss Landscape",
        show_minima: bool = True,
        height: int = 700,
        width: int = 900,
    ) -> go.Figure:
        """
        Create a single 3D view of the loss landscape with trajectories.

        Args:
            landscape: Computed landscape data
            paths: List of optimization paths to overlay
            title: Plot title
            show_minima: Whether to show known minima markers
            height: Figure height in pixels
            width: Figure width in pixels

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Add surface
        fig.add_trace(self.create_surface(landscape))

        # Add trajectories
        if paths:
            for path in paths:
                for trace in self.create_trajectory(path, z_offset=0.1):
                    fig.add_trace(trace)

        # Add minima markers
        if show_minima:
            for trace in self.create_minima_markers(landscape.function_name, landscape):
                fig.add_trace(trace)

        # Configure layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20),
                x=0.5,
                xanchor="center",
            ),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Loss",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1),
                ),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.8),
            ),
            height=height,
            width=width,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
            ),
        )

        return fig

    def create_comparison_view(
        self,
        landscape: LandscapeData,
        paths_dict: Dict[str, OptimizationPath],
        title: str = "Optimizer Comparison",
        height: int = 600,
        width: int = 1400,
    ) -> go.Figure:
        """
        Create side-by-side 3D views comparing multiple optimizers.

        Each optimizer gets its own subplot with the same landscape,
        making it easy to compare their trajectories.

        Args:
            landscape: Computed landscape data
            paths_dict: Dictionary mapping optimizer names to paths
            title: Overall figure title
            height: Figure height in pixels
            width: Figure width in pixels

        Returns:
            Plotly Figure with subplots
        """
        n_optimizers = len(paths_dict)

        if n_optimizers == 0:
            return go.Figure()

        # Create subplot layout
        cols = min(n_optimizers, 3)
        rows = (n_optimizers + cols - 1) // cols

        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "surface"} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=list(paths_dict.keys()),
            horizontal_spacing=0.02,
            vertical_spacing=0.08,
        )

        # Add surface and trajectory to each subplot
        for idx, (opt_name, path) in enumerate(paths_dict.items()):
            row = idx // cols + 1
            col = idx % cols + 1

            # Add surface (lower opacity for clarity)
            surface = self.create_surface(landscape, opacity=0.7)
            fig.add_trace(surface, row=row, col=col)

            # Add trajectory
            for trace in self.create_trajectory(path, z_offset=0.1):
                fig.add_trace(trace, row=row, col=col)

            # Add minima markers
            for trace in self.create_minima_markers(landscape.function_name, landscape):
                fig.add_trace(trace, row=row, col=col)

        # Update all scene layouts
        for i in range(1, n_optimizers + 1):
            scene_name = f"scene{i}" if i > 1 else "scene"
            fig.update_layout(
                **{
                    scene_name: dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Loss",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                        aspectmode="manual",
                        aspectratio=dict(x=1, y=1, z=0.8),
                    )
                }
            )

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24),
                x=0.5,
                xanchor="center",
            ),
            height=height * rows,
            width=width,
            showlegend=False,
            margin=dict(l=20, r=20, t=100, b=20),
        )

        return fig

    def create_2d_contour_view(
        self,
        landscape: LandscapeData,
        paths: Optional[List[OptimizationPath]] = None,
        title: str = "Loss Landscape (Top View)",
        height: int = 600,
        width: int = 700,
    ) -> go.Figure:
        """
        Create a 2D contour plot with trajectory overlays.

        Useful for examining the path details without 3D rotation.

        Args:
            landscape: Computed landscape data
            paths: List of optimization paths to overlay
            title: Plot title
            height: Figure height in pixels
            width: Figure width in pixels

        Returns:
            Plotly Figure with contour plot
        """
        z_values = landscape.z_values.copy()

        if self.config.log_scale_z:
            z_values = np.log1p(z_values - z_values.min() + 1)

        fig = go.Figure()

        # Add contour plot
        fig.add_trace(
            go.Contour(
                x=landscape.x_grid[0],
                y=landscape.y_grid[:, 0],
                z=z_values,
                colorscale=self.config.surface_colorscale,
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=10, color="white"),
                ),
                colorbar=dict(title="Loss"),
                hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>loss: %{z:.4f}<extra></extra>",
            )
        )

        # Add trajectories
        if paths:
            for path in paths:
                # Main trajectory
                fig.add_trace(
                    go.Scatter(
                        x=path.x_coords,
                        y=path.y_coords,
                        mode="lines+markers",
                        line=dict(color=path.color, width=3),
                        marker=dict(size=4, color=path.color),
                        name=path.optimizer_name,
                        hovertemplate=(
                            f"<b>{path.optimizer_name}</b><br>"
                            "x: %{x:.3f}<br>"
                            "y: %{y:.3f}<extra></extra>"
                        ),
                    )
                )

                # Start marker
                if len(path.x_coords) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[path.x_coords[0]],
                            y=[path.y_coords[0]],
                            mode="markers",
                            marker=dict(
                                size=12,
                                color="white",
                                symbol="circle",
                                line=dict(color=path.color, width=3),
                            ),
                            showlegend=False,
                            hovertemplate=f"<b>{path.optimizer_name} Start</b><extra></extra>",
                        )
                    )

                    # End marker
                    fig.add_trace(
                        go.Scatter(
                            x=[path.x_coords[-1]],
                            y=[path.y_coords[-1]],
                            mode="markers",
                            marker=dict(
                                size=10,
                                color=path.color,
                                symbol="diamond",
                            ),
                            showlegend=False,
                            hovertemplate=f"<b>{path.optimizer_name} End</b><extra></extra>",
                        )
                    )

        # Add minima markers
        if landscape.function_name in FUNCTION_REGISTRY:
            func_info = FUNCTION_REGISTRY[landscape.function_name]
            for mx, my in func_info.get("minima", []):
                fig.add_trace(
                    go.Scatter(
                        x=[mx],
                        y=[my],
                        mode="markers",
                        marker=dict(
                            size=15,
                            color="gold",
                            symbol="star",
                            line=dict(color="black", width=1),
                        ),
                        name="Minimum",
                        hovertemplate=f"<b>Minimum</b><br>({mx:.3f}, {my:.3f})<extra></extra>",
                    )
                )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18), x=0.5, xanchor="center"),
            xaxis_title="X",
            yaxis_title="Y",
            height=height,
            width=width,
            xaxis=dict(scaleanchor="y", scaleratio=1),
        )

        return fig

    def create_animation(
        self,
        landscape: LandscapeData,
        paths: List[OptimizationPath],
        title: str = "Optimization Animation",
        frame_duration: int = 50,
        height: int = 700,
        width: int = 900,
    ) -> go.Figure:
        """
        Create an animated visualization of optimization paths.

        Args:
            landscape: Computed landscape data
            paths: List of optimization paths to animate
            title: Plot title
            frame_duration: Duration of each frame in milliseconds
            height: Figure height
            width: Figure width

        Returns:
            Plotly Figure with animation frames
        """
        # Find max steps across all paths
        max_steps = max(len(p.steps) for p in paths)

        # Create base figure with surface
        fig = go.Figure()
        fig.add_trace(self.create_surface(landscape))

        # Add minima markers
        for trace in self.create_minima_markers(landscape.function_name, landscape):
            fig.add_trace(trace)

        # Create frames
        frames = []
        for step in range(1, max_steps + 1):
            frame_data = [self.create_surface(landscape)]

            for path in paths:
                n_points = min(step, len(path.steps))
                x = path.x_coords[:n_points]
                y = path.y_coords[:n_points]
                z = path.z_coords[:n_points]

                if self.config.log_scale_z:
                    z = np.log1p(z - path.z_coords.min() + 1)
                z = z * self.config.z_scale + 0.1

                # Trajectory line
                frame_data.append(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="lines+markers",
                        line=dict(color=path.color, width=4),
                        marker=dict(size=4, color=path.color),
                        name=path.optimizer_name,
                    )
                )

                # Current position marker
                if len(x) > 0:
                    frame_data.append(
                        go.Scatter3d(
                            x=[x[-1]],
                            y=[y[-1]],
                            z=[z[-1]],
                            mode="markers",
                            marker=dict(size=10, color=path.color, symbol="circle"),
                            showlegend=False,
                        )
                    )

            frames.append(go.Frame(data=frame_data, name=str(step)))

        fig.frames = frames

        # Add play/pause buttons
        fig.update_layout(
            title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Loss",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.8),
            ),
            height=height,
            width=width,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.1,
                    xanchor="left",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=frame_duration, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix="Step: ",
                        visible=True,
                        xanchor="right",
                    ),
                    transition=dict(duration=0),
                    pad=dict(b=10, t=50),
                    len=0.8,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[
                                [str(k)],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=str(k),
                            method="animate",
                        )
                        for k in range(1, max_steps + 1)
                    ],
                )
            ],
        )

        return fig


def create_stats_table(paths: Dict[str, OptimizationPath]) -> go.Figure:
    """
    Create a summary statistics table for optimizer comparison.

    Args:
        paths: Dictionary of optimization paths

    Returns:
        Plotly Table figure
    """
    headers = ["Optimizer", "Steps", "Final Loss", "Final X", "Final Y", "Converged"]

    values = [
        [p.optimizer_name for p in paths.values()],
        [len(p.steps) for p in paths.values()],
        [f"{p.final_loss:.6f}" for p in paths.values()],
        [f"{p.x_coords[-1]:.4f}" if len(p.x_coords) > 0 else "N/A" for p in paths.values()],
        [f"{p.y_coords[-1]:.4f}" if len(p.y_coords) > 0 else "N/A" for p in paths.values()],
        ["✓" if p.converged else "✗" for p in paths.values()],
    ]

    colors = [p.color for p in paths.values()]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="rgb(50, 50, 50)",
                    font=dict(color="white", size=14),
                    align="center",
                ),
                cells=dict(
                    values=values,
                    fill_color=[
                        [f"rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, 0.3)" for c in colors]
                        for _ in headers
                    ],
                    font=dict(size=12),
                    align="center",
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(text="Optimization Results", font=dict(size=16)),
        height=200,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


if __name__ == "__main__":
    from landscape import LossLandscape, LandscapeConfig
    from optimizer_tracker import OptimizerTracker

    # Create landscape and paths for testing
    landscape_computer = LossLandscape(LandscapeConfig(grid_size=80))
    landscape = landscape_computer.compute_function_landscape("rosenbrock")

    tracker = OptimizerTracker()
    paths = tracker.compare_optimizers(
        function_name="rosenbrock",
        optimizer_names=["sgd", "adam", "rmsprop"],
        start_point=(-1.5, 1.5),
        num_steps=150,
    )

    # Create visualizer and test different views
    viz = LossLandscapeVisualizer()

    # Single view
    fig1 = viz.create_single_view(
        landscape,
        list(paths.values()),
        title="Rosenbrock Function - Optimizer Comparison",
    )

    # Comparison view
    fig2 = viz.create_comparison_view(landscape, paths)

    # 2D contour view
    fig3 = viz.create_2d_contour_view(landscape, list(paths.values()))

    # Animation
    fig4 = viz.create_animation(landscape, list(paths.values()))

    print("Visualization tests complete. Figures created successfully.")
    print(f"- Single view: {len(fig1.data)} traces")
    print(f"- Comparison view: {len(fig2.data)} traces")
    print(f"- Contour view: {len(fig3.data)} traces")
    print(f"- Animation: {len(fig4.frames)} frames")
