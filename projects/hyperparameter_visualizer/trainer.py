"""
Training utilities with real-time callbacks for visualization.

Provides a Trainer class that supports:
- Multiple optimizers (SGD, Adam, AdamW, RMSprop)
- Learning rate schedulers
- Real-time loss and accuracy tracking
- Callback system for UI updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field
import time


@dataclass
class TrainingMetrics:
    """Container for training metrics updated in real-time."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    current_epoch: int = 0
    total_epochs: int = 0
    is_training: bool = False
    should_stop: bool = False

    def reset(self):
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_accuracies.clear()
        self.val_accuracies.clear()
        self.learning_rates.clear()
        self.epoch_times.clear()
        self.current_epoch = 0
        self.is_training = False
        self.should_stop = False


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9
) -> torch.optim.Optimizer:
    """
    Create optimizer by name.

    Args:
        model: Neural network model
        optimizer_name: One of 'sgd', 'adam', 'adamw', 'rmsprop'
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        momentum: Momentum (for SGD and RMSprop)

    Returns:
        PyTorch optimizer
    """
    params = model.parameters()

    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            params, lr=learning_rate,
            momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "adam":
        return torch.optim.Adam(
            params, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            params, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "rmsprop":
        return torch.optim.RMSprop(
            params, lr=learning_rate,
            momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    num_epochs: int,
    **kwargs
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Create learning rate scheduler by name.

    Args:
        optimizer: PyTorch optimizer
        scheduler_name: One of 'none', 'step', 'cosine', 'exponential'
        num_epochs: Total number of training epochs

    Returns:
        PyTorch scheduler or None
    """
    if scheduler_name.lower() == "none":
        return None
    elif scheduler_name.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, num_epochs // 3), gamma=0.5
        )
    elif scheduler_name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    elif scheduler_name.lower() == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95
        )
    elif scheduler_name.lower() == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class Trainer:
    """
    Trainer with real-time callback support.

    Supports streaming metrics to UI during training.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.metrics = TrainingMetrics()

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module
    ) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item() * X.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def run_validation(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> tuple[float, float]:
        """Run validation on the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in val_loader:
            X, y = X.to(self.device), y.to(self.device)

            outputs = self.model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item() * X.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        callback: Optional[Callable[[TrainingMetrics], None]] = None
    ) -> TrainingMetrics:
        """
        Full training loop with callbacks.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            callback: Function called after each epoch with metrics

        Returns:
            Final training metrics
        """
        criterion = nn.CrossEntropyLoss()

        self.metrics.reset()
        self.metrics.total_epochs = num_epochs
        self.metrics.is_training = True

        for epoch in range(num_epochs):
            if self.metrics.should_stop:
                break

            self.metrics.current_epoch = epoch + 1
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion)

            # Validate
            val_loss, val_acc = self.run_validation(val_loader, criterion)

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Record metrics
            self.metrics.train_losses.append(train_loss)
            self.metrics.val_losses.append(val_loss)
            self.metrics.train_accuracies.append(train_acc)
            self.metrics.val_accuracies.append(val_acc)
            self.metrics.learning_rates.append(current_lr)
            self.metrics.epoch_times.append(epoch_time)

            # Callback for UI update
            if callback is not None:
                callback(self.metrics)

        self.metrics.is_training = False
        return self.metrics

    def stop_training(self):
        """Signal to stop training early."""
        self.metrics.should_stop = True


def run_training(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 0.01,
    batch_size: int = 32,
    num_epochs: int = 50,
    weight_decay: float = 0.0,
    scheduler_name: str = "none",
    callback: Optional[Callable[[TrainingMetrics], None]] = None
) -> TrainingMetrics:
    """
    Convenience function to run complete training.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model: Neural network model
        optimizer_name: Optimizer to use
        learning_rate: Initial learning rate
        batch_size: Batch size
        num_epochs: Number of epochs
        weight_decay: L2 regularization
        scheduler_name: LR scheduler to use
        callback: Progress callback

    Returns:
        Training metrics
    """
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, optimizer_name, learning_rate, weight_decay)
    scheduler = create_scheduler(optimizer, scheduler_name, num_epochs)

    # Create trainer and run
    trainer = Trainer(model, optimizer, scheduler)
    return trainer.train(train_loader, val_loader, num_epochs, callback)


# Test when run directly
if __name__ == "__main__":
    from neural_network import SimpleMLP, generate_spiral_data

    print("Testing trainer...")

    # Generate data
    X, y = generate_spiral_data(n_samples=500)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Create model
    model = SimpleMLP(input_size=2, hidden_sizes=(32, 32), output_size=2)

    # Define callback
    def print_progress(metrics: TrainingMetrics):
        print(f"Epoch {metrics.current_epoch}/{metrics.total_epochs} - "
              f"Train Loss: {metrics.train_losses[-1]:.4f}, "
              f"Val Acc: {metrics.val_accuracies[-1]:.2%}")

    # Train
    metrics = run_training(
        X_train, y_train, X_val, y_val,
        model=model,
        optimizer_name="adam",
        learning_rate=0.01,
        batch_size=32,
        num_epochs=20,
        callback=print_progress
    )

    print(f"\nFinal Val Accuracy: {metrics.val_accuracies[-1]:.2%}")
    print("Trainer test passed!")
