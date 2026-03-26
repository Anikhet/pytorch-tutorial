"""
Training and knowledge distillation for edge image classification.

This module handles two training workflows:
1. Standard supervised training of the teacher model on MNIST.
2. Knowledge distillation from teacher to student, where the student
   learns from the teacher's soft predictions (logits / temperature)
   combined with the hard ground-truth labels.

Usage:
    python trainer.py

This will train the teacher, distill the student, and save both
checkpoints to disk.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import TeacherCNN, StudentCNN, get_model_info


def get_data_loaders(batch_size=64):
    """
    Create MNIST train and test data loaders.

    Args:
        batch_size: Number of samples per batch.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train_model(model, train_loader, epochs=3, lr=0.001):
    """
    Train a model with standard cross-entropy loss.

    Creates an internal copy so the original model is not mutated.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.

    Returns:
        A new trained model (deep copy of the input).
    """
    trained = copy.deepcopy(model)
    optimizer = torch.optim.Adam(trained.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    trained.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = trained(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return trained


def test_accuracy(model, test_loader):
    """
    Evaluate classification accuracy on a test set.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    model_copy = copy.deepcopy(model)
    model_copy.train(False)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model_copy(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total


def distill_model(teacher, student, train_loader, epochs=3, temperature=4.0, alpha=0.7):
    """
    Train a student model using knowledge distillation from a teacher.

    The loss is a weighted combination of:
    - KL divergence between teacher and student soft predictions (scaled by T^2)
    - Standard cross-entropy with ground-truth labels

    Neither the teacher nor the original student is mutated.

    Args:
        teacher: Trained teacher model (frozen during distillation).
        student: Untrained student model to be distilled.
        train_loader: DataLoader for training data.
        epochs: Number of distillation epochs.
        temperature: Softmax temperature for softer probability distributions.
        alpha: Weight for the distillation loss (1-alpha for hard label loss).

    Returns:
        A new trained student model (deep copy).
    """
    # Work on copies to avoid mutating inputs
    teacher_copy = copy.deepcopy(teacher)
    trained_student = copy.deepcopy(student)

    teacher_copy.train(False)
    trained_student.train()
    optimizer = torch.optim.Adam(trained_student.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()

            # Teacher predictions (no gradient needed)
            with torch.no_grad():
                teacher_logits = teacher_copy(images)

            student_logits = trained_student(images)

            # Soft targets: KL divergence on temperature-scaled softmax
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)

            # Hard targets: standard cross-entropy
            hard_loss = F.cross_entropy(student_logits, labels)

            # Combined loss
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch + 1}/{epochs} - Distillation Loss: {avg_loss:.4f}")

    return trained_student


if __name__ == "__main__":
    print("=" * 50)
    print("Edge Image Classifier - Training Pipeline")
    print("=" * 50)

    train_loader, test_loader = get_data_loaders()

    # --- Train Teacher ---
    print("\n[1/2] Training Teacher CNN...")
    teacher = TeacherCNN()
    trained_teacher = train_model(teacher, train_loader, epochs=3)
    teacher_acc = test_accuracy(trained_teacher, test_loader)
    teacher_info = get_model_info(trained_teacher)
    print(f"  Teacher accuracy: {teacher_acc:.2%}")
    print(f"  Teacher params: {teacher_info['param_count']:,} ({teacher_info['size_kb']:.1f} KB)")

    # --- Distill Student ---
    print("\n[2/2] Distilling Student CNN from Teacher...")
    student = StudentCNN()
    trained_student = distill_model(trained_teacher, student, train_loader, epochs=3)
    student_acc = test_accuracy(trained_student, test_loader)
    student_info = get_model_info(trained_student)
    print(f"  Student accuracy: {student_acc:.2%}")
    print(f"  Student params: {student_info['param_count']:,} ({student_info['size_kb']:.1f} KB)")

    # --- Save checkpoints ---
    torch.save(trained_teacher.state_dict(), "teacher.pth")
    torch.save(trained_student.state_dict(), "student.pth")
    print("\nSaved: teacher.pth, student.pth")

    # --- Summary ---
    print("\n" + "-" * 50)
    print(f"{'Model':<20} {'Params':>10} {'Size (KB)':>10} {'Accuracy':>10}")
    print("-" * 50)
    print(f"{'Teacher':<20} {teacher_info['param_count']:>10,} {teacher_info['size_kb']:>10.1f} {teacher_acc:>10.2%}")
    print(f"{'Student':<20} {student_info['param_count']:>10,} {student_info['size_kb']:>10.1f} {student_acc:>10.2%}")
    print("-" * 50)
