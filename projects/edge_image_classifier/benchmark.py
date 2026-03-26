"""
Performance benchmarking across all pipeline stages.

Measures inference latency for each model variant:
- Teacher (original large model)
- Student (distilled)
- Student (quantized)
- Student (ONNX runtime)

Prints a comprehensive comparison table with size, accuracy, and speed.

Usage:
    python benchmark.py

Expects trainer.py and export.py to have been run first.
"""

import copy
import os
import time

import numpy as np
import onnxruntime as ort
import torch

from model import TeacherCNN, StudentCNN, get_model_info
from trainer import get_data_loaders, test_accuracy
from compress import quantize_model, get_file_size


def benchmark_pytorch(model, input_shape=(1, 1, 28, 28), num_runs=100):
    """
    Measure average inference time for a PyTorch model.

    Runs the model multiple times on random input and returns the
    average time per forward pass in milliseconds.

    Args:
        model: A PyTorch model.
        input_shape: Shape of the input tensor.
        num_runs: Number of inference runs to average over.

    Returns:
        Average inference time in milliseconds.
    """
    model_copy = copy.deepcopy(model)
    model_copy.train(False)
    dummy = torch.randn(*input_shape)

    # Warmup runs to stabilize timing
    with torch.no_grad():
        for _ in range(10):
            model_copy(dummy)

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            model_copy(dummy)
    elapsed = time.perf_counter() - start

    return (elapsed / num_runs) * 1000  # Convert to ms


def benchmark_onnx(onnx_path, input_shape=(1, 1, 28, 28), num_runs=100):
    """
    Measure average inference time for an ONNX model.

    Uses ONNX Runtime for inference. Runs the model multiple times
    and returns the average time per forward pass in milliseconds.

    Args:
        onnx_path: Path to the .onnx model file.
        input_shape: Shape of the input tensor.
        num_runs: Number of inference runs to average over.

    Returns:
        Average inference time in milliseconds.
    """
    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Warmup runs
    for _ in range(10):
        session.run(None, {"input": dummy})

    # Timed runs
    start = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, {"input": dummy})
    elapsed = time.perf_counter() - start

    return (elapsed / num_runs) * 1000  # Convert to ms


def _check_file_exists(path, source_script):
    """Check if a required file exists and print an error if not."""
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run {source_script} first.")
        raise SystemExit(1)


if __name__ == "__main__":
    print("=" * 65)
    print("Edge Image Classifier - Benchmark")
    print("=" * 65)

    # Verify prerequisites
    _check_file_exists("teacher.pth", "trainer.py")
    _check_file_exists("student.pth", "trainer.py")
    _check_file_exists("student.onnx", "export.py")

    _, test_loader = get_data_loaders()

    # --- Load models ---
    teacher = TeacherCNN()
    teacher.load_state_dict(torch.load("teacher.pth", weights_only=True))

    student = StudentCNN()
    student.load_state_dict(torch.load("student.pth", weights_only=True))

    quantized_student = quantize_model(student)

    # --- Benchmark all variants ---
    print("\nBenchmarking (100 runs each)...\n")

    teacher_ms = benchmark_pytorch(teacher)
    student_ms = benchmark_pytorch(student)
    quantized_ms = benchmark_pytorch(quantized_student)
    onnx_ms = benchmark_onnx("student.onnx")

    # --- Gather metrics ---
    teacher_acc = test_accuracy(teacher, test_loader)
    student_acc = test_accuracy(student, test_loader)
    quantized_acc = test_accuracy(quantized_student, test_loader)

    teacher_size = get_file_size(teacher)
    student_size = get_file_size(student)
    quantized_size = get_file_size(quantized_student)
    onnx_size = os.path.getsize("student.onnx")

    # --- Print comparison table ---
    print("-" * 65)
    print(
        f"{'Model':<22} {'Size':>8} {'Accuracy':>10} "
        f"{'Latency':>10} {'Speedup':>8}"
    )
    print("-" * 65)

    rows = [
        ("Teacher", teacher_size, teacher_acc, teacher_ms),
        ("Student (distilled)", student_size, student_acc, student_ms),
        ("Student (quantized)", quantized_size, quantized_acc, quantized_ms),
        ("Student (ONNX)", onnx_size, student_acc, onnx_ms),
    ]

    baseline_ms = teacher_ms
    for name, size, acc, latency in rows:
        size_str = f"{size / 1024:.1f} KB"
        speedup = baseline_ms / max(latency, 1e-6)
        print(
            f"{name:<22} {size_str:>8} {acc:>10.2%} "
            f"{latency:>8.2f}ms {speedup:>7.1f}x"
        )

    print("-" * 65)
    print("\nCompression ratio (teacher -> quantized):", end=" ")
    print(f"{teacher_size / max(quantized_size, 1):.1f}x smaller")
    print("Done.")
