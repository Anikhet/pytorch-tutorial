"""
Training monitor with detailed metric collection.

Provides real-time tracking of:
- Loss curves (train/val)
- Accuracy metrics
- Gradient statistics per layer
- Weight distributions
- Learning rate schedules
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Callable, Optional, Dict, List, Any
from dataclasses import dataclass, field
import time
import queue

from neural_network import ModelStats


@dataclass
class TrainingSnapshot:
    """Single snapshot of training state."""
    epoch: int
    batch: int
    train_loss: float
    train_acc: float
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    learning_rate: float = 0.0
    model_stats: Optional[ModelStats] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingHistory:
    """Complete training history."""
    snapshots: List[TrainingSnapshot] = field(default_factory=list)
    epoch_train_losses: List[float] = field(default_factory=list)
    epoch_val_losses: List[float] = field(default_factory=list)
    epoch_train_accs: List[float] = field(default_factory=list)
    epoch_val_accs: List[float] = field(default_factory=list)
    epoch_learning_rates: List[float] = field(default_factory=list)
    batch_losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    current_epoch: int = 0
    total_epochs: int = 0
    is_training: bool = False
    should_stop: bool = False

    def reset(self):
        self.snapshots.clear()
        self.epoch_train_losses.clear()
        self.epoch_val_losses.clear()
        self.epoch_train_accs.clear()
        self.epoch_val_accs.clear()
        self.epoch_learning_rates.clear()
        self.batch_losses.clear()
        self.grad_norms.clear()
        self.current_epoch = 0
        self.is_training = False
        self.should_stop = False


class TrainingMonitor:
    """
    Training monitor with detailed metric collection.

    Tracks and stores metrics for dashboard visualization.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cpu",
        collect_every_n_batches: int = 5
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.collect_every_n_batches = collect_every_n_batches

        self.history = TrainingHistory()
        self.update_queue: queue.Queue = queue.Queue()

        # Register hooks if model supports it
        if hasattr(model, 'register_hooks'):
            model.register_hooks()

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        epoch: int
    ) -> tuple[float, float]:
        """Train for one epoch with detailed tracking."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()

            # Collect gradient norm before clipping
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            self.history.grad_norms.append(grad_norm)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Track batch loss
            batch_loss = loss.item()
            self.history.batch_losses.append(batch_loss)
            total_loss += batch_loss * X.size(0)

            # Accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

            # Collect detailed stats periodically
            if batch_idx % self.collect_every_n_batches == 0:
                model_stats = None
                if hasattr(self.model, 'get_model_stats'):
                    model_stats = self.model.get_model_stats()

                snapshot = TrainingSnapshot(
                    epoch=epoch,
                    batch=batch_idx,
                    train_loss=batch_loss,
                    train_acc=correct / total,
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    model_stats=model_stats
                )
                self.history.snapshots.append(snapshot)

                # Put update in queue for dashboard
                self.update_queue.put(snapshot)

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
        callback: Optional[Callable[[TrainingHistory], None]] = None
    ) -> TrainingHistory:
        """
        Full training loop with monitoring.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            callback: Called after each epoch

        Returns:
            Complete training history
        """
        criterion = nn.CrossEntropyLoss()

        self.history.reset()
        self.history.total_epochs = num_epochs
        self.history.is_training = True

        for epoch in range(1, num_epochs + 1):
            if self.history.should_stop:
                break

            self.history.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, epoch)

            # Validate
            val_loss, val_acc = self.run_validation(val_loader, criterion)

            # Record epoch metrics
            self.history.epoch_train_losses.append(train_loss)
            self.history.epoch_val_losses.append(val_loss)
            self.history.epoch_train_accs.append(train_acc)
            self.history.epoch_val_accs.append(val_acc)
            self.history.epoch_learning_rates.append(self.optimizer.param_groups[0]['lr'])

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Callback
            if callback is not None:
                callback(self.history)

        self.history.is_training = False
        return self.history

    def stop_training(self):
        """Signal to stop training."""
        self.history.should_stop = True

    def get_latest_snapshot(self) -> Optional[TrainingSnapshot]:
        """Get the most recent training snapshot."""
        if self.history.snapshots:
            return self.history.snapshots[-1]
        return None

    def get_gradient_stats_df(self) -> Dict[str, List]:
        """Get gradient statistics as a dictionary for DataFrame."""
        if not self.history.snapshots:
            return {}

        latest = self.history.snapshots[-1]
        if latest.model_stats is None:
            return {}

        data = {
            "Layer": [],
            "Grad Mean": [],
            "Grad Std": [],
            "Grad Norm": [],
            "Weight Mean": [],
            "Weight Std": []
        }

        for name, stats in latest.model_stats.layer_stats.items():
            data["Layer"].append(name)
            data["Grad Mean"].append(f"{stats.grad_mean:.6f}")
            data["Grad Std"].append(f"{stats.grad_std:.6f}")
            data["Grad Norm"].append(f"{stats.grad_norm:.4f}")
            data["Weight Mean"].append(f"{stats.weight_mean:.6f}")
            data["Weight Std"].append(f"{stats.weight_std:.6f}")

        return data


def create_mnist_loaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """Create MNIST-like synthetic data loaders for testing."""
    n_train, n_val = 5000, 1000

    X_train = torch.randn(n_train, 784)
    y_train = torch.randint(0, 10, (n_train,))

    X_val = torch.randn(n_val, 784)
    y_val = torch.randint(0, 10, (n_val,))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def create_spiral_loaders(batch_size: int = 32) -> tuple[DataLoader, DataLoader]:
    """Create spiral classification data loaders."""
    n_samples = 1000
    n_classes = 3

    X, y = [], []
    samples_per_class = n_samples // n_classes

    for class_idx in range(n_classes):
        r = np.linspace(0.2, 1, samples_per_class)
        theta = np.linspace(
            class_idx * 4, (class_idx + 1) * 4, samples_per_class
        ) + np.random.randn(samples_per_class) * 0.2

        X.append(np.column_stack([r * np.sin(theta), r * np.cos(theta)]))
        y.append(np.full(samples_per_class, class_idx))

    X = np.vstack(X)
    y = np.concatenate(y)

    # Shuffle and split
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, y_train = torch.FloatTensor(X[:split]), torch.LongTensor(y[:split])
    X_val, y_val = torch.FloatTensor(X[split:]), torch.LongTensor(y[split:])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Test when run directly
if __name__ == "__main__":
    from neural_network import MonitoredMLP

    print("Testing training monitor...")

    # Create model and data
    model = MonitoredMLP(input_size=2, hidden_sizes=(32, 16), output_size=3)
    train_loader, val_loader = create_spiral_loaders(batch_size=32)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Create monitor
    monitor = TrainingMonitor(model, optimizer, collect_every_n_batches=5)

    # Train with callback
    def print_progress(history: TrainingHistory):
        print(f"Epoch {history.current_epoch}/{history.total_epochs} - "
              f"Train Loss: {history.epoch_train_losses[-1]:.4f}, "
              f"Val Acc: {history.epoch_val_accs[-1]:.2%}")

    history = monitor.train(train_loader, val_loader, num_epochs=10, callback=print_progress)

    print(f"\nTotal snapshots: {len(history.snapshots)}")
    print(f"Total batch losses: {len(history.batch_losses)}")
    print(f"Final val accuracy: {history.epoch_val_accs[-1]:.2%}")

    # Test gradient stats
    grad_stats = monitor.get_gradient_stats_df()
    print(f"\nGradient stats layers: {grad_stats.get('Layer', [])}")

    print("\nTraining monitor test passed!")
