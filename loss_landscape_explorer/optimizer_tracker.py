"""
Optimizer Path Tracker Module

Tracks and records the optimization trajectory of various PyTorch optimizers
(SGD, Adam, RMSprop, etc.) as they navigate a loss landscape.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Type
import numpy as np
import torch
import torch.optim as optim

from landscape import create_pytorch_loss_function, FUNCTION_REGISTRY


@dataclass
class OptimizerConfig:
    """Configuration for an optimizer."""
    name: str
    optimizer_class: Type[optim.Optimizer]
    lr: float = 0.01
    kwargs: Dict = field(default_factory=dict)
    color: str = "#FF6B6B"
    display_name: Optional[str] = None

    @property
    def label(self) -> str:
        """Get display label for the optimizer."""
        return self.display_name or self.name


@dataclass
class OptimizationStep:
    """Single step in the optimization trajectory."""
    step: int
    x: float
    y: float
    loss: float
    grad_x: float
    grad_y: float


@dataclass
class OptimizationPath:
    """Complete optimization trajectory for one optimizer."""
    optimizer_name: str
    color: str
    steps: List[OptimizationStep]
    final_loss: float
    converged: bool

    @property
    def x_coords(self) -> np.ndarray:
        """Get x coordinates as numpy array."""
        return np.array([s.x for s in self.steps])

    @property
    def y_coords(self) -> np.ndarray:
        """Get y coordinates as numpy array."""
        return np.array([s.y for s in self.steps])

    @property
    def z_coords(self) -> np.ndarray:
        """Get loss values as numpy array."""
        return np.array([s.loss for s in self.steps])

    @property
    def trajectory(self) -> np.ndarray:
        """Get full trajectory as (N, 3) array."""
        return np.column_stack([self.x_coords, self.y_coords, self.z_coords])


# Pre-configured optimizer configurations with distinct colors
OPTIMIZER_PRESETS = {
    "sgd": OptimizerConfig(
        name="sgd",
        optimizer_class=optim.SGD,
        lr=0.01,
        color="#FF6B6B",
        display_name="SGD",
    ),
    "sgd_momentum": OptimizerConfig(
        name="sgd_momentum",
        optimizer_class=optim.SGD,
        lr=0.01,
        kwargs={"momentum": 0.9},
        color="#FF8E53",
        display_name="SGD + Momentum",
    ),
    "sgd_nesterov": OptimizerConfig(
        name="sgd_nesterov",
        optimizer_class=optim.SGD,
        lr=0.01,
        kwargs={"momentum": 0.9, "nesterov": True},
        color="#FFA726",
        display_name="SGD + Nesterov",
    ),
    "adam": OptimizerConfig(
        name="adam",
        optimizer_class=optim.Adam,
        lr=0.1,
        color="#4ECDC4",
        display_name="Adam",
    ),
    "adamw": OptimizerConfig(
        name="adamw",
        optimizer_class=optim.AdamW,
        lr=0.1,
        color="#45B7D1",
        display_name="AdamW",
    ),
    "rmsprop": OptimizerConfig(
        name="rmsprop",
        optimizer_class=optim.RMSprop,
        lr=0.01,
        color="#96CEB4",
        display_name="RMSprop",
    ),
    "adagrad": OptimizerConfig(
        name="adagrad",
        optimizer_class=optim.Adagrad,
        lr=0.5,
        color="#DDA0DD",
        display_name="Adagrad",
    ),
    "adadelta": OptimizerConfig(
        name="adadelta",
        optimizer_class=optim.Adadelta,
        lr=1.0,
        color="#FFB6C1",
        display_name="Adadelta",
    ),
}


class OptimizerTracker:
    """
    Tracks optimization paths across a loss landscape.

    Records the position, loss, and gradient at each step for multiple
    optimizers, enabling side-by-side comparison of their trajectories.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the optimizer tracker.

        Args:
            device: Device to run optimization on
        """
        self.device = device

    def optimize(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        optimizer_config: OptimizerConfig,
        start_point: Tuple[float, float],
        num_steps: int = 100,
        convergence_threshold: float = 1e-6,
    ) -> OptimizationPath:
        """
        Run optimization and track the path.

        Args:
            loss_fn: Loss function taking a 2D parameter tensor
            optimizer_config: Configuration for the optimizer
            start_point: Starting (x, y) coordinates
            num_steps: Maximum number of optimization steps
            convergence_threshold: Stop if loss change is below this

        Returns:
            OptimizationPath containing the full trajectory
        """
        # Initialize parameters at start point
        params = torch.tensor(
            [start_point[0], start_point[1]],
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )

        # Create optimizer
        optimizer = optimizer_config.optimizer_class(
            [params],
            lr=optimizer_config.lr,
            **optimizer_config.kwargs,
        )

        steps: List[OptimizationStep] = []
        prev_loss = float("inf")
        converged = False

        for step_num in range(num_steps):
            optimizer.zero_grad()

            # Compute loss and gradients
            loss = loss_fn(params)
            loss.backward()

            # Record step
            step = OptimizationStep(
                step=step_num,
                x=params[0].item(),
                y=params[1].item(),
                loss=loss.item(),
                grad_x=params.grad[0].item() if params.grad is not None else 0.0,
                grad_y=params.grad[1].item() if params.grad is not None else 0.0,
            )
            steps.append(step)

            # Check for convergence
            loss_change = abs(prev_loss - loss.item())
            if loss_change < convergence_threshold and step_num > 10:
                converged = True
                break

            prev_loss = loss.item()

            # Update parameters
            optimizer.step()

        return OptimizationPath(
            optimizer_name=optimizer_config.label,
            color=optimizer_config.color,
            steps=steps,
            final_loss=steps[-1].loss if steps else float("inf"),
            converged=converged,
        )

    def compare_optimizers(
        self,
        function_name: str,
        optimizer_names: List[str],
        start_point: Tuple[float, float],
        num_steps: int = 100,
        learning_rates: Optional[Dict[str, float]] = None,
    ) -> Dict[str, OptimizationPath]:
        """
        Compare multiple optimizers on the same loss landscape.

        Args:
            function_name: Name of loss function from FUNCTION_REGISTRY
            optimizer_names: List of optimizer names from OPTIMIZER_PRESETS
            start_point: Starting (x, y) coordinates for all optimizers
            num_steps: Maximum number of optimization steps
            learning_rates: Optional dict of custom learning rates per optimizer

        Returns:
            Dictionary mapping optimizer names to their paths
        """
        loss_fn = create_pytorch_loss_function(function_name)
        learning_rates = learning_rates or {}

        paths = {}
        for opt_name in optimizer_names:
            if opt_name not in OPTIMIZER_PRESETS:
                available = ", ".join(OPTIMIZER_PRESETS.keys())
                raise ValueError(f"Unknown optimizer: {opt_name}. Available: {available}")

            config = OPTIMIZER_PRESETS[opt_name]

            # Apply custom learning rate if provided
            if opt_name in learning_rates:
                config = OptimizerConfig(
                    name=config.name,
                    optimizer_class=config.optimizer_class,
                    lr=learning_rates[opt_name],
                    kwargs=config.kwargs.copy(),
                    color=config.color,
                    display_name=config.display_name,
                )

            path = self.optimize(
                loss_fn=loss_fn,
                optimizer_config=config,
                start_point=start_point,
                num_steps=num_steps,
            )
            paths[opt_name] = path

        return paths

    def run_multiple_starts(
        self,
        function_name: str,
        optimizer_name: str,
        start_points: List[Tuple[float, float]],
        num_steps: int = 100,
    ) -> List[OptimizationPath]:
        """
        Run optimizer from multiple starting points.

        Useful for visualizing how optimizer behavior depends on initialization.

        Args:
            function_name: Name of loss function
            optimizer_name: Name of optimizer to use
            start_points: List of (x, y) starting coordinates
            num_steps: Maximum steps per run

        Returns:
            List of OptimizationPaths, one per starting point
        """
        loss_fn = create_pytorch_loss_function(function_name)
        config = OPTIMIZER_PRESETS[optimizer_name]

        paths = []
        for start in start_points:
            path = self.optimize(
                loss_fn=loss_fn,
                optimizer_config=config,
                start_point=start,
                num_steps=num_steps,
            )
            paths.append(path)

        return paths


def get_recommended_starts(function_name: str) -> List[Tuple[float, float]]:
    """
    Get recommended starting points for a given function.

    These starting points are chosen to demonstrate interesting
    optimizer behavior for each loss function.

    Args:
        function_name: Name of the loss function

    Returns:
        List of recommended (x, y) starting coordinates
    """
    recommendations = {
        "rosenbrock": [(-1.5, 1.5), (-0.5, -0.5), (1.5, -1.0)],
        "rastrigin": [(4.0, 4.0), (-3.0, 3.0), (2.0, -2.0)],
        "beale": [(-4.0, -4.0), (4.0, 4.0), (0.0, 0.0)],
        "himmelblau": [(0.0, 0.0), (-4.0, -4.0), (4.0, 4.0)],
        "saddle": [(1.0, 1.0), (-1.0, -1.0), (1.5, -0.5)],
        "ackley": [(4.0, 4.0), (-3.0, -3.0), (2.0, -2.0)],
        "goldstein_price": [(-1.5, -1.5), (1.5, 0.5), (0.0, 0.0)],
    }
    return recommendations.get(function_name, [(0.0, 0.0)])


def get_function_optimal_lr(function_name: str, optimizer_name: str) -> float:
    """
    Get a recommended learning rate for a function-optimizer pair.

    These values are tuned to produce good visualization results
    (neither too slow nor overshooting).

    Args:
        function_name: Name of the loss function
        optimizer_name: Name of the optimizer

    Returns:
        Recommended learning rate
    """
    # Tuned learning rates for good visualization
    lr_map = {
        "rosenbrock": {
            "sgd": 0.001,
            "sgd_momentum": 0.001,
            "adam": 0.05,
            "rmsprop": 0.01,
            "adagrad": 0.3,
        },
        "rastrigin": {
            "sgd": 0.01,
            "sgd_momentum": 0.005,
            "adam": 0.1,
            "rmsprop": 0.01,
            "adagrad": 0.5,
        },
        "beale": {
            "sgd": 0.0001,
            "sgd_momentum": 0.0001,
            "adam": 0.1,
            "rmsprop": 0.01,
            "adagrad": 0.3,
        },
        "himmelblau": {
            "sgd": 0.01,
            "sgd_momentum": 0.005,
            "adam": 0.1,
            "rmsprop": 0.02,
            "adagrad": 0.5,
        },
        "saddle": {
            "sgd": 0.05,
            "sgd_momentum": 0.02,
            "adam": 0.1,
            "rmsprop": 0.05,
            "adagrad": 0.5,
        },
        "ackley": {
            "sgd": 0.05,
            "sgd_momentum": 0.02,
            "adam": 0.1,
            "rmsprop": 0.02,
            "adagrad": 0.5,
        },
        "goldstein_price": {
            "sgd": 0.00001,
            "sgd_momentum": 0.00001,
            "adam": 0.05,
            "rmsprop": 0.001,
            "adagrad": 0.1,
        },
    }

    default = OPTIMIZER_PRESETS.get(optimizer_name, OPTIMIZER_PRESETS["sgd"]).lr
    return lr_map.get(function_name, {}).get(optimizer_name, default)


if __name__ == "__main__":
    # Test the optimizer tracker
    tracker = OptimizerTracker()

    # Compare optimizers on Rosenbrock function
    print("Comparing optimizers on Rosenbrock function:")
    print("-" * 50)

    paths = tracker.compare_optimizers(
        function_name="rosenbrock",
        optimizer_names=["sgd", "adam", "rmsprop"],
        start_point=(-1.5, 1.5),
        num_steps=200,
    )

    for opt_name, path in paths.items():
        print(
            f"{path.optimizer_name:15s} | "
            f"Steps: {len(path.steps):4d} | "
            f"Final Loss: {path.final_loss:12.6f} | "
            f"Converged: {path.converged}"
        )

    # Test multiple starting points
    print("\nAdam from multiple starting points on Himmelblau:")
    print("-" * 50)

    multi_paths = tracker.run_multiple_starts(
        function_name="himmelblau",
        optimizer_name="adam",
        start_points=[(0.0, 0.0), (-4.0, -4.0), (4.0, 4.0), (-4.0, 4.0)],
        num_steps=100,
    )

    for i, path in enumerate(multi_paths):
        end_x, end_y = path.x_coords[-1], path.y_coords[-1]
        print(f"Start {i+1}: Final position ({end_x:.3f}, {end_y:.3f}), Loss: {path.final_loss:.6f}")
