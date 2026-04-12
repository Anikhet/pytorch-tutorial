"""
ONNX export and verification for edge deployment.

This module exports a trained PyTorch model to ONNX format and verifies
that the exported model produces the same outputs as the original.
ONNX models can run on any compatible runtime (mobile, web, embedded).

Usage:
    python export.py

Expects 'student.pth' to exist (produced by trainer.py).
"""

import copy
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import StudentCNN


def export_to_onnx(model, filepath, input_shape=(1, 1, 28, 28)):
    """
    Export a PyTorch model to ONNX format.

    Creates a deep copy so the original model is not mutated.
    The exported model is validated using the ONNX checker.

    Args:
        model: A trained PyTorch model.
        filepath: Output path for the .onnx file.
        input_shape: Tuple specifying the input tensor shape.
            Defaults to a single MNIST image (1, 1, 28, 28).

    Returns:
        The filepath of the exported ONNX model.
    """
    model_copy = copy.deepcopy(model)
    model_copy.train(False)

    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model_copy,
        dummy_input,
        filepath,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )

    # Validate the exported model
    onnx_model = onnx.load(filepath)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX model exported and validated: {filepath}")

    return filepath


def verify_onnx(onnx_path, pytorch_model, input_shape=(1, 1, 28, 28)):
    """
    Verify that ONNX model outputs match PyTorch model outputs.

    Runs the same random input through both models and checks that
    the outputs are numerically close (within a tolerance of 1e-5).

    Args:
        onnx_path: Path to the exported .onnx file.
        pytorch_model: The original PyTorch model for comparison.
        input_shape: Tuple specifying the input tensor shape.

    Returns:
        True if outputs match within tolerance, False otherwise.
    """
    model_copy = copy.deepcopy(pytorch_model)
    model_copy.train(False)

    # Generate a deterministic test input
    torch.manual_seed(42)
    test_input = torch.randn(*input_shape)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = model_copy(test_input).numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(
        None,
        {"input": test_input.numpy()},
    )[0]

    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    matches = max_diff < 1e-5

    if matches:
        print(f"  Verification PASSED (max diff: {max_diff:.2e})")
    else:
        print(f"  Verification FAILED (max diff: {max_diff:.2e})")

    return matches


if __name__ == "__main__":
    print("=" * 50)
    print("Edge Image Classifier - ONNX Export")
    print("=" * 50)

    # Load trained student
    student = StudentCNN()
    if not os.path.exists("student.pth"):
        print("Error: student.pth not found. Run trainer.py first.")
        raise SystemExit(1)

    student.load_state_dict(torch.load("student.pth", weights_only=True))

    # Export to ONNX
    print("\n[1/2] Exporting to ONNX...")
    onnx_path = export_to_onnx(student, "student.onnx")

    # Verify correctness
    print("\n[2/2] Verifying ONNX model...")
    verify_onnx(onnx_path, student)

    # Report file sizes
    pytorch_size = os.path.getsize("student.pth")
    onnx_size = os.path.getsize("student.onnx")
    print(f"\n  PyTorch size: {pytorch_size / 1024:.1f} KB")
    print(f"  ONNX size:    {onnx_size / 1024:.1f} KB")
    print("\nDone. ONNX model ready for deployment.")
