"""
Compression pipeline for edge deployment: pruning + quantization.

This module applies two compression techniques sequentially:
1. Unstructured L1 pruning on convolutional layers to zero out
   the least important weights.
2. Dynamic quantization on linear layers to convert FP32 weights
   to INT8, reducing model size by roughly 4x.

The pipeline takes a trained student model and produces a compressed
version suitable for edge devices.

Usage:
    python compress.py

Expects 'student.pth' to exist (produced by trainer.py).
"""

import copy
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from model import StudentCNN, get_model_info
from trainer import get_data_loaders, test_accuracy


def prune_model(model, amount=0.3):
    """
    Apply unstructured L1 pruning to all Conv2d layers.

    Creates a deep copy so the original model is not mutated.
    Pruning zeroes out the smallest-magnitude weights, making
    the model sparser without changing the architecture.

    Args:
        model: A trained PyTorch model.
        amount: Fraction of weights to prune (0.0 to 1.0).

    Returns:
        A new pruned model (deep copy with pruning applied).
    """
    pruned = copy.deepcopy(model)

    # Collect all Conv2d layers for pruning
    layers_to_prune = [
        (module, "weight")
        for module in pruned.modules()
        if isinstance(module, nn.Conv2d)
    ]

    # Apply global unstructured L1 pruning across all conv layers
    # This prunes the globally least-important weights rather than
    # pruning each layer independently
    prune.global_unstructured(
        layers_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Make pruning permanent by removing the reparameterization
    for module, param_name in layers_to_prune:
        prune.remove(module, param_name)

    return pruned


def quantize_model(model):
    """
    Apply dynamic quantization to all Linear layers.

    Dynamic quantization converts FP32 weights to INT8 and performs
    computation in INT8 where possible. This reduces model size and
    can speed up inference on CPU.

    Creates a deep copy so the original model is not mutated.

    Args:
        model: A trained PyTorch model.

    Returns:
        A new dynamically quantized model.
    """
    model_copy = copy.deepcopy(model)
    model_copy.train(False)

    quantized = torch.quantization.quantize_dynamic(
        model_copy,
        {nn.Linear},  # Quantize all Linear layers
        dtype=torch.qint8,
    )
    return quantized


def get_file_size(model):
    """
    Measure the serialized size of a model in bytes.

    Saves the model's state_dict to a temporary file, reads its size,
    then cleans up the temporary file.

    Args:
        model: A PyTorch model.

    Returns:
        Size in bytes as an integer.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp_path = tmp.name
        torch.save(model.state_dict(), tmp_path)

    size = os.path.getsize(tmp_path)
    os.unlink(tmp_path)
    return size


def compress_pipeline(model):
    """
    Apply the full compression pipeline: prune then quantize.

    Creates a deep copy so the original model is not mutated.

    Args:
        model: A trained PyTorch model.

    Returns:
        A dict with keys:
        - 'pruned': The pruned model.
        - 'quantized': The pruned + quantized model.
    """
    pruned = prune_model(model, amount=0.3)
    quantized = quantize_model(pruned)
    return {
        "pruned": pruned,
        "quantized": quantized,
    }


def _format_size(size_bytes):
    """Format byte count as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    return f"{size_bytes / 1024:.1f} KB"


if __name__ == "__main__":
    print("=" * 55)
    print("Edge Image Classifier - Compression Pipeline")
    print("=" * 55)

    # Load the trained student model
    student = StudentCNN()
    if not os.path.exists("student.pth"):
        print("Error: student.pth not found. Run trainer.py first.")
        raise SystemExit(1)

    student.load_state_dict(torch.load("student.pth", weights_only=True))
    _, test_loader = get_data_loaders()

    # --- Baseline ---
    baseline_acc = test_accuracy(student, test_loader)
    baseline_size = get_file_size(student)
    print(f"\nBaseline Student:")
    print(f"  Accuracy: {baseline_acc:.2%}")
    print(f"  Size:     {_format_size(baseline_size)}")

    # --- Compression ---
    print("\nApplying compression pipeline...")
    result = compress_pipeline(student)

    pruned = result["pruned"]
    quantized = result["quantized"]

    pruned_acc = test_accuracy(pruned, test_loader)
    pruned_size = get_file_size(pruned)
    print(f"\nAfter Pruning (30% sparsity):")
    print(f"  Accuracy: {pruned_acc:.2%}")
    print(f"  Size:     {_format_size(pruned_size)}")

    quantized_size = get_file_size(quantized)
    quantized_acc = test_accuracy(quantized, test_loader)
    print(f"\nAfter Pruning + Quantization:")
    print(f"  Accuracy: {quantized_acc:.2%}")
    print(f"  Size:     {_format_size(quantized_size)}")

    # --- Save compressed model ---
    torch.save(pruned.state_dict(), "student_pruned.pth")
    torch.save(quantized.state_dict(), "student_compressed.pth")
    print("\nSaved: student_pruned.pth, student_compressed.pth")

    # --- Summary Table ---
    print("\n" + "-" * 55)
    print(f"{'Stage':<25} {'Size':>10} {'Accuracy':>10} {'Reduction':>10}")
    print("-" * 55)
    print(
        f"{'Student (baseline)':<25} "
        f"{_format_size(baseline_size):>10} "
        f"{baseline_acc:>10.2%} "
        f"{'1.0x':>10}"
    )
    print(
        f"{'Pruned (30%)':<25} "
        f"{_format_size(pruned_size):>10} "
        f"{pruned_acc:>10.2%} "
        f"{baseline_size / max(pruned_size, 1):>9.1f}x"
    )
    print(
        f"{'Pruned + Quantized':<25} "
        f"{_format_size(quantized_size):>10} "
        f"{quantized_acc:>10.2%} "
        f"{baseline_size / max(quantized_size, 1):>9.1f}x"
    )
    print("-" * 55)
