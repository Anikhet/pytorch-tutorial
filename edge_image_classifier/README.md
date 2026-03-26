# Edge Image Classifier

A beginner-friendly project that demonstrates the full edge ML compression pipeline.

## What This Project Does

Takes a trained image classifier and makes it small enough for edge devices through:

1. **Knowledge Distillation** (teacher -> student)
2. **Pruning** (remove unimportant weights)
3. **Quantization** (FP32 -> INT8)
4. **Export to ONNX** format

## Pipeline Overview

```
Teacher CNN (large, accurate)
    |
    v  Knowledge Distillation
Student CNN (small, nearly as accurate)
    |
    v  Pruning
Student CNN (sparse, slightly smaller)
    |
    v  Dynamic Quantization
Student CNN (INT8 weights, much smaller)
    |
    v  ONNX Export
Portable model for any runtime
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Train teacher and distill student
python trainer.py

# Step 2: Apply pruning + quantization
python compress.py

# Step 3: Export to ONNX
python export.py

# Step 4: Compare all stages
python benchmark.py
```

## Project Structure

| File           | Description                                |
|----------------|--------------------------------------------|
| `model.py`     | Model definitions (TeacherCNN, StudentCNN) |
| `trainer.py`   | Training and distillation logic            |
| `compress.py`  | Pruning and quantization pipeline          |
| `export.py`    | ONNX export and verification               |
| `benchmark.py` | Performance comparison across all stages   |

## Key Concepts

### Knowledge Distillation
The student model learns from the teacher's soft predictions (logits)
rather than just hard labels. This transfers "dark knowledge" -- the
teacher's understanding of which classes look similar to each other.

### Pruning
Removes weights closest to zero, making the model sparser. We use
unstructured L1 pruning on convolutional layers.

### Dynamic Quantization
Converts FP32 weights to INT8 at inference time, reducing model size
by ~4x with minimal accuracy loss.

### ONNX Export
Converts the PyTorch model to ONNX format, enabling deployment on
any runtime that supports ONNX (mobile, web, embedded).

## Expected Results

After running the full pipeline, you should see something like:

| Stage                | Size   | Accuracy | Speedup |
|----------------------|--------|----------|---------|
| Teacher              | ~2MB   | ~99%     | 1x      |
| Student (distilled)  | ~200KB | ~98%     | ~3x     |
| Pruned + Quantized   | ~60KB  | ~97%     | ~5x     |

*Results vary by hardware and random seed.*

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CPU only (no GPU required)

## Learning Resources

- [PyTorch Knowledge Distillation](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Export](https://pytorch.org/docs/stable/onnx.html)
