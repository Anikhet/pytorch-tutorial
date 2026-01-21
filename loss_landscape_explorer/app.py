"""
Loss Landscape Explorer - Streamlit App

Interactive 3D visualization of loss landscapes with optimizer path comparison.
Compare SGD, Adam, RMSprop and other optimizers side-by-side. Zoom into
saddle points and local minima. Watch optimizer trajectories as glowing paths.
"""

import streamlit as st
import numpy as np

from landscape import (
    LossLandscape,
    LandscapeConfig,
    FUNCTION_REGISTRY,
)
from optimizer_tracker import (
    OptimizerTracker,
    OPTIMIZER_PRESETS,
    get_recommended_starts,
    get_function_optimal_lr,
)
from visualizer import (
    LossLandscapeVisualizer,
    VisualizerConfig,
    create_stats_table,
)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Loss Landscape Explorer",
        page_icon="🏔️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏔️ Loss Landscape Explorer")
    st.markdown(
        """
        Visualize loss landscapes and compare optimizer trajectories in 3D.
        Watch SGD, Adam, and RMSprop navigate through valleys, saddle points, and local minima.
        """
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Function selection
        st.subheader("Loss Function")
        function_name = st.selectbox(
            "Select function",
            options=list(FUNCTION_REGISTRY.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
            help="Choose a test function to visualize",
        )

        func_info = FUNCTION_REGISTRY[function_name]
        st.caption(func_info["description"])

        # Display known minima
        if func_info.get("minima"):
            minima_str = ", ".join([f"({m[0]}, {m[1]})" for m in func_info["minima"]])
            st.info(f"📍 Minima: {minima_str}")

        st.divider()

        # Optimizer selection
        st.subheader("Optimizers")
        selected_optimizers = st.multiselect(
            "Select optimizers to compare",
            options=list(OPTIMIZER_PRESETS.keys()),
            default=["sgd", "adam", "rmsprop"],
            format_func=lambda x: OPTIMIZER_PRESETS[x].display_name,
            help="Choose 1-6 optimizers to compare",
        )

        if len(selected_optimizers) == 0:
            st.warning("Please select at least one optimizer")
            selected_optimizers = ["sgd"]

        st.divider()

        # Starting point configuration
        st.subheader("Starting Point")
        recommended_starts = get_recommended_starts(function_name)

        start_mode = st.radio(
            "Start point selection",
            options=["recommended", "custom"],
            format_func=lambda x: "Recommended" if x == "recommended" else "Custom",
        )

        if start_mode == "recommended":
            start_idx = st.selectbox(
                "Select starting point",
                options=range(len(recommended_starts)),
                format_func=lambda i: f"({recommended_starts[i][0]:.1f}, {recommended_starts[i][1]:.1f})",
            )
            start_point = recommended_starts[start_idx]
        else:
            default_range = func_info["range"]
            col1, col2 = st.columns(2)
            with col1:
                start_x = st.number_input(
                    "Start X",
                    value=float(default_range[0] + 0.5),
                    min_value=float(default_range[0]),
                    max_value=float(default_range[1]),
                    step=0.1,
                )
            with col2:
                start_y = st.number_input(
                    "Start Y",
                    value=float(default_range[1] - 0.5),
                    min_value=float(default_range[0]),
                    max_value=float(default_range[1]),
                    step=0.1,
                )
            start_point = (start_x, start_y)

        st.divider()

        # Optimization settings
        st.subheader("Optimization")
        num_steps = st.slider(
            "Number of steps",
            min_value=10,
            max_value=500,
            value=150,
            step=10,
            help="Maximum optimization steps",
        )

        use_auto_lr = st.checkbox(
            "Use auto-tuned learning rates",
            value=True,
            help="Use pre-tuned learning rates for better visualization",
        )

        if not use_auto_lr:
            st.caption("Custom learning rates:")
            custom_lrs = {}
            for opt_name in selected_optimizers:
                default_lr = OPTIMIZER_PRESETS[opt_name].lr
                custom_lrs[opt_name] = st.number_input(
                    f"{OPTIMIZER_PRESETS[opt_name].display_name} LR",
                    value=default_lr,
                    min_value=0.0001,
                    max_value=10.0,
                    format="%.4f",
                    key=f"lr_{opt_name}",
                )
        else:
            custom_lrs = {
                opt: get_function_optimal_lr(function_name, opt)
                for opt in selected_optimizers
            }

        st.divider()

        # Visualization settings
        st.subheader("Visualization")
        grid_size = st.slider(
            "Surface resolution",
            min_value=30,
            max_value=150,
            value=80,
            step=10,
            help="Higher = smoother but slower",
        )

        colorscale = st.selectbox(
            "Color scheme",
            options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Greens"],
            index=0,
        )

        surface_opacity = st.slider(
            "Surface opacity",
            min_value=0.3,
            max_value=1.0,
            value=0.85,
            step=0.05,
        )

        log_scale = st.checkbox(
            "Log scale Z-axis",
            value=False,
            help="Use log scale for better visibility of steep functions",
        )

        show_minima = st.checkbox("Show known minima", value=True)

    # Main content area
    col_left, col_right = st.columns([3, 1])

    # Run optimization
    with st.spinner("Computing loss landscape and optimizer paths..."):
        # Create landscape
        landscape_config = LandscapeConfig(grid_size=grid_size)
        landscape_computer = LossLandscape(landscape_config)
        landscape = landscape_computer.compute_function_landscape(function_name)

        # Run optimizers
        tracker = OptimizerTracker()
        paths = tracker.compare_optimizers(
            function_name=function_name,
            optimizer_names=selected_optimizers,
            start_point=start_point,
            num_steps=num_steps,
            learning_rates=custom_lrs,
        )

        # Create visualizer
        viz_config = VisualizerConfig(
            surface_opacity=surface_opacity,
            surface_colorscale=colorscale,
            log_scale_z=log_scale,
        )
        visualizer = LossLandscapeVisualizer(viz_config)

    # Display results
    with col_left:
        # View selector
        view_mode = st.radio(
            "View mode",
            options=["combined", "side_by_side", "contour", "animated"],
            format_func=lambda x: {
                "combined": "🏔️ Combined 3D View",
                "side_by_side": "📊 Side-by-Side Comparison",
                "contour": "🗺️ 2D Contour View",
                "animated": "🎬 Animated",
            }[x],
            horizontal=True,
        )

        if view_mode == "combined":
            fig = visualizer.create_single_view(
                landscape,
                list(paths.values()),
                title=f"{function_name.replace('_', ' ').title()} - Optimizer Comparison",
                show_minima=show_minima,
                height=700,
                width=None,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif view_mode == "side_by_side":
            fig = visualizer.create_comparison_view(
                landscape,
                paths,
                title=f"{function_name.replace('_', ' ').title()} - Side-by-Side",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif view_mode == "contour":
            fig = visualizer.create_2d_contour_view(
                landscape,
                list(paths.values()),
                title=f"{function_name.replace('_', ' ').title()} - Contour View",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif view_mode == "animated":
            st.info("Use the Play/Pause buttons and slider below the plot to control the animation.")
            fig = visualizer.create_animation(
                landscape,
                list(paths.values()),
                title=f"{function_name.replace('_', ' ').title()} - Animation",
                frame_duration=80,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📈 Results")

        # Summary statistics
        for opt_name, path in paths.items():
            with st.expander(f"**{path.optimizer_name}**", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Steps", len(path.steps))
                    st.metric("Final Loss", f"{path.final_loss:.4f}")
                with col2:
                    st.metric("Final X", f"{path.x_coords[-1]:.3f}")
                    st.metric("Final Y", f"{path.y_coords[-1]:.3f}")

                if path.converged:
                    st.success("✓ Converged")
                else:
                    st.warning("⚡ Still optimizing")

                # Show learning rate used
                st.caption(f"LR: {custom_lrs.get(opt_name, OPTIMIZER_PRESETS[opt_name].lr):.4f}")

        st.divider()

        # Quick insights
        st.subheader("💡 Insights")

        # Find best performer
        best_opt = min(paths.items(), key=lambda x: x[1].final_loss)
        st.markdown(f"**Lowest loss:** {best_opt[1].optimizer_name}")

        fastest_opt = min(paths.items(), key=lambda x: len(x[1].steps) if x[1].converged else float("inf"))
        if fastest_opt[1].converged:
            st.markdown(f"**Fastest convergence:** {fastest_opt[1].optimizer_name}")

        # Distance to nearest minimum
        if func_info.get("minima"):
            st.markdown("**Distance to minimum:**")
            for opt_name, path in paths.items():
                min_dist = min(
                    np.sqrt((path.x_coords[-1] - m[0]) ** 2 + (path.y_coords[-1] - m[1]) ** 2)
                    for m in func_info["minima"]
                )
                st.caption(f"{path.optimizer_name}: {min_dist:.4f}")

    # Footer with tips
    st.divider()
    with st.expander("📚 Tips & Explanation"):
        st.markdown(
            """
            ### Understanding the Visualization

            **Loss Functions:**
            - **Rosenbrock**: Banana-shaped valley - tests ability to follow narrow curved paths
            - **Rastrigin**: Many local minima - tests escaping local minima
            - **Himmelblau**: Four equal minima - tests initialization sensitivity
            - **Saddle**: Simple saddle point - tests escaping saddle points
            - **Beale/Ackley**: Flat regions with steep walls - tests navigating plateaus

            **Optimizer Behaviors:**
            - **SGD**: Basic gradient descent - can get stuck in local minima
            - **SGD + Momentum**: Helps escape local minima and speeds up convergence
            - **Adam**: Adaptive learning rates - generally robust across functions
            - **RMSprop**: Good for non-stationary objectives
            - **Adagrad**: Adapts LR per-parameter - good for sparse gradients

            **Markers:**
            - ⭐ Gold star: Known global/local minima
            - ❌ Red X: Saddle points
            - ⚪ White circle: Starting point
            - 💎 Diamond: Final position

            **Tips:**
            - Use log scale for functions with large value ranges
            - Try different starting points to see initialization effects
            - Compare adaptive (Adam) vs non-adaptive (SGD) optimizers
            - Watch how momentum helps escape saddle points
            """
        )


if __name__ == "__main__":
    main()
