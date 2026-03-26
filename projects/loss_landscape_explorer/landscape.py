"""
Loss Landscape Computation Module

Computes 2D loss landscapes for neural networks by varying weights along
two directions (random or PCA-based). Includes common test functions
like Rosenbrock, Rastrigin, and Beale for demonstrating optimizer behavior.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


@dataclass
class LandscapeConfig:
    """Configuration for loss landscape computation."""
    grid_size: int = 50
    range_min: float = -2.0
    range_max: float = 2.0
    device: str = "cpu"


@dataclass
class LandscapeData:
    """Container for computed loss landscape data."""
    x_grid: np.ndarray
    y_grid: np.ndarray
    z_values: np.ndarray
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    function_name: str


class TestFunctions:
    """
    Collection of test functions for optimizer visualization.

    These functions have known properties (minima, saddle points, etc.)
    that make them ideal for demonstrating optimizer behavior.
    """

    @staticmethod
    def rosenbrock(x: np.ndarray, y: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
        """
        Rosenbrock function - banana-shaped valley with global minimum at (a, a^2).

        Famous for testing optimization algorithms due to its narrow curved valley.
        The global minimum is at (1, 1) for default parameters.

        Args:
            x: X coordinates
            y: Y coordinates
            a: First parameter (default 1.0)
            b: Second parameter (default 100.0)

        Returns:
            Function values at each (x, y) point
        """
        return (a - x) ** 2 + b * (y - x ** 2) ** 2

    @staticmethod
    def rastrigin(x: np.ndarray, y: np.ndarray, A: float = 10.0) -> np.ndarray:
        """
        Rastrigin function - highly multimodal with many local minima.

        Global minimum at origin (0, 0) surrounded by many local minima.
        Excellent for testing global optimization capabilities.

        Args:
            x: X coordinates
            y: Y coordinates
            A: Amplitude parameter (default 10.0)

        Returns:
            Function values at each (x, y) point
        """
        return (
            2 * A
            + (x ** 2 - A * np.cos(2 * np.pi * x))
            + (y ** 2 - A * np.cos(2 * np.pi * y))
        )

    @staticmethod
    def beale(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Beale function - has a flat region and steep walls.

        Global minimum at (3, 0.5). Tests optimizer behavior in
        regions with varying curvature.

        Args:
            x: X coordinates
            y: Y coordinates

        Returns:
            Function values at each (x, y) point
        """
        term1 = (1.5 - x + x * y) ** 2
        term2 = (2.25 - x + x * y ** 2) ** 2
        term3 = (2.625 - x + x * y ** 3) ** 2
        return term1 + term2 + term3

    @staticmethod
    def himmelblau(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Himmelblau function - four identical local minima.

        Has minima at (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848).
        Useful for testing optimizer convergence to different basins.

        Args:
            x: X coordinates
            y: Y coordinates

        Returns:
            Function values at each (x, y) point
        """
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    @staticmethod
    def saddle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Simple saddle function (hyperbolic paraboloid).

        Saddle point at origin. Perfect for demonstrating
        optimizer behavior at saddle points.

        Args:
            x: X coordinates
            y: Y coordinates

        Returns:
            Function values at each (x, y) point
        """
        return x ** 2 - y ** 2

    @staticmethod
    def ackley(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Ackley function - nearly flat outer region with large hole at center.

        Global minimum at origin. Tests optimizer's ability to
        navigate flat regions to find the global minimum.

        Args:
            x: X coordinates
            y: Y coordinates

        Returns:
            Function values at each (x, y) point
        """
        term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
        term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        return term1 + term2 + np.e + 20

    @staticmethod
    def goldstein_price(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Goldstein-Price function - complex landscape with one global minimum.

        Global minimum at (0, -1) with value 3. Has several local minima
        and flat regions that challenge optimizers.

        Args:
            x: X coordinates
            y: Y coordinates

        Returns:
            Function values at each (x, y) point
        """
        term1 = 1 + ((x + y + 1) ** 2) * (
            19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2
        )
        term2 = 30 + ((2 * x - 3 * y) ** 2) * (
            18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2
        )
        return term1 * term2


# Mapping of function names to their implementations and metadata
FUNCTION_REGISTRY = {
    "rosenbrock": {
        "func": TestFunctions.rosenbrock,
        "range": (-2.0, 2.0),
        "minima": [(1.0, 1.0)],
        "description": "Banana-shaped valley, tests narrow path following",
    },
    "rastrigin": {
        "func": TestFunctions.rastrigin,
        "range": (-5.12, 5.12),
        "minima": [(0.0, 0.0)],
        "description": "Highly multimodal, many local minima",
    },
    "beale": {
        "func": TestFunctions.beale,
        "range": (-4.5, 4.5),
        "minima": [(3.0, 0.5)],
        "description": "Flat regions with steep walls",
    },
    "himmelblau": {
        "func": TestFunctions.himmelblau,
        "range": (-5.0, 5.0),
        "minima": [(3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)],
        "description": "Four identical local minima",
    },
    "saddle": {
        "func": TestFunctions.saddle,
        "range": (-2.0, 2.0),
        "minima": [],
        "saddle_points": [(0.0, 0.0)],
        "description": "Simple saddle point at origin",
    },
    "ackley": {
        "func": TestFunctions.ackley,
        "range": (-5.0, 5.0),
        "minima": [(0.0, 0.0)],
        "description": "Flat outer region with central hole",
    },
    "goldstein_price": {
        "func": TestFunctions.goldstein_price,
        "range": (-2.0, 2.0),
        "minima": [(0.0, -1.0)],
        "description": "Complex landscape with flat regions",
    },
}


class LossLandscape:
    """
    Computes loss landscapes for visualization.

    Supports both analytical test functions and neural network loss surfaces.
    """

    def __init__(self, config: Optional[LandscapeConfig] = None):
        """
        Initialize the loss landscape computer.

        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or LandscapeConfig()

    def compute_function_landscape(
        self,
        function_name: str,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
    ) -> LandscapeData:
        """
        Compute loss landscape for a named test function.

        Args:
            function_name: Name of function from FUNCTION_REGISTRY
            x_range: Custom x-axis range (uses function default if None)
            y_range: Custom y-axis range (uses function default if None)

        Returns:
            LandscapeData containing the computed surface

        Raises:
            ValueError: If function_name is not in registry
        """
        if function_name not in FUNCTION_REGISTRY:
            available = ", ".join(FUNCTION_REGISTRY.keys())
            raise ValueError(f"Unknown function: {function_name}. Available: {available}")

        func_info = FUNCTION_REGISTRY[function_name]
        func = func_info["func"]
        default_range = func_info["range"]

        x_range = x_range or default_range
        y_range = y_range or default_range

        return self._compute_landscape(func, x_range, y_range, function_name)

    def compute_custom_landscape(
        self,
        loss_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        name: str = "custom",
    ) -> LandscapeData:
        """
        Compute loss landscape for a custom function.

        Args:
            loss_func: Function taking (x, y) arrays and returning loss values
            x_range: Range for x-axis
            y_range: Range for y-axis
            name: Name for the function

        Returns:
            LandscapeData containing the computed surface
        """
        return self._compute_landscape(loss_func, x_range, y_range, name)

    def _compute_landscape(
        self,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        name: str,
    ) -> LandscapeData:
        """
        Internal method to compute the landscape grid.

        Args:
            func: Loss function
            x_range: X-axis range
            y_range: Y-axis range
            name: Function name

        Returns:
            LandscapeData with computed values
        """
        x = np.linspace(x_range[0], x_range[1], self.config.grid_size)
        y = np.linspace(y_range[0], y_range[1], self.config.grid_size)
        x_grid, y_grid = np.meshgrid(x, y)

        z_values = func(x_grid, y_grid)

        return LandscapeData(
            x_grid=x_grid,
            y_grid=y_grid,
            z_values=z_values,
            x_range=x_range,
            y_range=y_range,
            function_name=name,
        )


def create_pytorch_loss_function(
    function_name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a PyTorch-compatible loss function from a named test function.

    This wraps the numpy-based test functions for use with PyTorch optimizers.

    Args:
        function_name: Name of function from FUNCTION_REGISTRY

    Returns:
        A function that takes a 2D tensor [x, y] and returns a scalar loss
    """
    if function_name not in FUNCTION_REGISTRY:
        available = ", ".join(FUNCTION_REGISTRY.keys())
        raise ValueError(f"Unknown function: {function_name}. Available: {available}")

    def loss_fn(params: torch.Tensor) -> torch.Tensor:
        x, y = params[0], params[1]

        if function_name == "rosenbrock":
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
        elif function_name == "rastrigin":
            A = 10.0
            return (
                2 * A
                + (x ** 2 - A * torch.cos(2 * np.pi * x))
                + (y ** 2 - A * torch.cos(2 * np.pi * y))
            )
        elif function_name == "beale":
            term1 = (1.5 - x + x * y) ** 2
            term2 = (2.25 - x + x * y ** 2) ** 2
            term3 = (2.625 - x + x * y ** 3) ** 2
            return term1 + term2 + term3
        elif function_name == "himmelblau":
            return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
        elif function_name == "saddle":
            return x ** 2 - y ** 2
        elif function_name == "ackley":
            term1 = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x ** 2 + y ** 2)))
            term2 = -torch.exp(0.5 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y)))
            return term1 + term2 + np.e + 20
        elif function_name == "goldstein_price":
            term1 = 1 + ((x + y + 1) ** 2) * (
                19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2
            )
            term2 = 30 + ((2 * x - 3 * y) ** 2) * (
                18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2
            )
            return term1 * term2
        else:
            raise ValueError(f"Unknown function: {function_name}")

    return loss_fn


if __name__ == "__main__":
    # Test the landscape computation
    landscape = LossLandscape(LandscapeConfig(grid_size=100))

    for func_name in FUNCTION_REGISTRY:
        data = landscape.compute_function_landscape(func_name)
        print(f"{func_name}: shape={data.z_values.shape}, range=[{data.z_values.min():.2f}, {data.z_values.max():.2f}]")

    # Test PyTorch loss function
    loss_fn = create_pytorch_loss_function("rosenbrock")
    params = torch.tensor([0.0, 0.0], requires_grad=True)
    loss = loss_fn(params)
    loss.backward()
    print(f"\nRosenbrock at (0, 0): loss={loss.item():.4f}, grad={params.grad.tolist()}")
