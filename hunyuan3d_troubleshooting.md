# Hunyuan3D-2 on RunPod: Issues & Fixes

A log of every issue encountered while setting up Tencent's Hunyuan3D-2 (image-to-3D with texture) on RunPod.

---

## 1. RTX 5090 Not Supported

**Error:** `NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible`

**Cause:** RTX 5090 (Blackwell architecture, sm_120) is not supported by any stable PyTorch release yet. Only nightly builds have partial support.

**Fix:** Switch to RTX 4090 (Ada Lovelace, sm_89) which is fully supported.

---

## 2. Root Disk Full (20 GB)

**Error:** `OSError: No space left on device` during model download

**Cause:** HuggingFace downloads model weights (~28 GB total) to `/root/.cache` by default, which lives on RunPod's container disk (only 20 GB).

**Fix:**
- Increase container disk to 40 GB and volume disk to 50 GB
- Symlink `/root/.cache` to `/workspace/.cache` (persistent volume)
- Set all HF environment variables (`HF_HOME`, `HF_HUB_CACHE`, etc.) to point to `/workspace`

---

## 3. RunPod Template Mislabeled (torch 2.10.0 nightly)

**Error:** `RuntimeError: Unknown operator aten::OpaqueObject`

**Cause:** The "RunPod Pytorch 2.4.0" template on some pods actually shipped `torch 2.10.0.dev` (a nightly build) instead of stable 2.4.0, causing operator incompatibilities.

**Fix:** Explicitly install a pinned stable PyTorch version instead of relying on the template:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

---

## 4. CUDA 12.1 Too Old for Texture Rasterizer

**Error:** C++ compilation errors when building `custom_rasterizer` and `differentiable_renderer`

**Cause:** The texture generation C++ extensions require CUDA 12.4+ to compile. The "PyTorch 2.2.0" template ships CUDA 12.1.

**Fix:** Use the "RunPod Pytorch 2.4.0" template which ships CUDA 12.4.1.

---

## 5. transformers CVE-2025-32434 Security Check

**Error:** `ValueError: This transformers version requires torch >= 2.6`

**Cause:** `transformers >= 4.48` enforces PyTorch >= 2.6 due to a `torch.load` security vulnerability (CVE-2025-32434). The project's `requirements.txt` pulls the latest transformers but doesn't pin torch.

**Fix:** Install torch 2.6.0+cu124 explicitly (satisfies both CUDA 12.4 and transformers requirements).

---

## 6. typing_extensions Missing `TypeIs`

**Error:** `ImportError: cannot import name 'TypeIs' from 'typing_extensions'`

**Cause:** PyTorch 2.6 requires `typing_extensions >= 4.10` for `TypeIs` support. The template's pre-installed version was too old.

**Fix:** Upgrade before installing torch:
```bash
pip install "typing_extensions>=4.10"
```

---

## 7. numpy / trimesh / scipy Incompatibility

**Error:** `AttributeError: module 'numpy._core.records' has no attribute '_byteorderconv'`

**Cause:** numpy 2.0 introduced breaking changes that trimesh and scipy haven't fully adapted to.

**Fix:** Pin numpy to <2.0:
```bash
pip install "numpy>=1.26.4,<2.0"
```

---

## 8. requirements.txt Downgrades torch

**Error:** After running `pip install -r requirements.txt`, torch gets replaced with an incompatible version.

**Cause:** The project's `requirements.txt` has zero version pins. pip resolves dependencies freely and may pull a different torch version.

**Fix:** Re-install torch 2.6.0+cu124 after the requirements.txt install:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

---

## 9. torchvision CUDA Index Mismatch

**Error:** torchvision installed from wrong CUDA index (cu128 instead of cu124), causing version conflicts.

**Cause:** Not specifying `--index-url` when installing torchvision separately causes pip to pull from the default PyPI index.

**Fix:** Always install torch, torchvision, and torchaudio together with the same `--index-url`:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

---

## 10. transformers `find_pruneable_heads_and_indices` ImportError

**Error:** `ImportError: cannot import name 'find_pruneable_heads_and_indices'`

**Cause:** Mixed old/new transformers files left in `site-packages` after partial upgrades.

**Fix:** Force reinstall:
```bash
pip uninstall transformers -y && pip install transformers --force-reinstall
```

---

## 11. pymeshlab `libOpenGL.so.0` Warnings

**Error:**
```
Cannot load library libfilter_ao.so: (libOpenGL.so.0: cannot open shared object file)
```

**Cause:** RunPod containers don't have OpenGL libraries installed. pymeshlab plugins that need OpenGL fail to load.

**Fix:** These are cosmetic warnings — they affect optional mesh filters but don't block generation. Can be fixed with:
```bash
apt-get install -y libopengl0
```

---

## 12. GPU Memory Leak Requiring Pod Restart

**Error:** `torch.cuda.memory_allocated()` shows 24 GB used with no active processes.

**Cause:** Failed pipeline runs or interrupted cells leave GPU memory allocated. Python garbage collection doesn't always free CUDA memory.

**Fix:** `torch.cuda.empty_cache()` helps partially. Full fix requires kernel restart. In severe cases, stop and restart the entire pod.

---

## 13. Texture Pipeline Runs on CPU (0% GPU Utilization)

**Error:** GPU utilization shows 0% during Step 6 (texture generation), CPU at 100%. Texture takes 10+ minutes.

**Cause:** `hy3dgen/texgen/pipelines.py` defaults `self.device = 'cpu'` in the texture config. The x4 upscaler and paint pipeline run entirely on CPU despite CUDA being available. This is a known issue (GitHub Issue #352, HuggingFace Discussion #6).

**Fix:** Patch the source file **before** importing the module:
```bash
sed -i "s/self.device = 'cpu'/self.device = 'cuda'/g" /workspace/Hunyuan3D-2/hy3dgen/texgen/pipelines.py
```
Then restart the kernel so Python picks up the patched file. Also override at runtime:
```python
paint_pipeline.device = 'cuda'
if hasattr(paint_pipeline, 'worker'):
    paint_pipeline.worker.device = 'cuda'
```

**Additional cause:** Python module caching. If Step 5 imports from `hy3dgen` (e.g., `hy3dgen.shapegen`), Python caches the `hy3dgen` package. When Step 6 later runs `sed` and imports `hy3dgen.texgen`, the patched file on disk is ignored — Python serves the cached module with `device='cpu'`. Also, setting `paint_pipeline.device = 'cuda'` only overrides a top-level attribute; internal sub-components (UNet, VAE, x4 upscaler) retain their own `device='cpu'` references.

**Better fix:** Clear cached modules before importing, then deep-override all sub-components:
```python
import sys
mods_to_remove = [key for key in sys.modules if 'hy3dgen.texgen' in key]
for mod in mods_to_remove:
    del sys.modules[mod]

from hy3dgen.texgen import Hunyuan3DPaintPipeline
paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")

# Deep override: move ALL nn.Module sub-components to CUDA
import torch
for attr_name in dir(paint_pipeline):
    try:
        attr = getattr(paint_pipeline, attr_name, None)
        if isinstance(attr, torch.nn.Module):
            attr.to('cuda')
    except Exception:
        pass
```

---

## 14. Missing Background Removal Step

**Error:** Poor 3D model quality — artifacts, floating geometry.

**Cause:** Passing a raw JPG (with background) directly to the shape pipeline. The model expects a transparent-background RGBA image.

**Fix:** Use the built-in `BackgroundRemover` as shown in the official examples:
```python
from hy3dgen.rembg import BackgroundRemover
raw_image = Image.open("input.jpg")
image = raw_image.convert("RGBA")
if raw_image.mode == "RGB":
    rembg = BackgroundRemover()
    image = rembg(image)
```

---

## Summary: Recommended RunPod Configuration

| Setting | Value |
|---------|-------|
| GPU | L40S (48 GB VRAM, 16 vCPU) or RTX 4090 (24 GB) |
| Template | RunPod Pytorch 2.4.0 (CUDA 12.4.1) |
| Container Disk | 40 GB |
| Volume Disk | 50 GB |
| PyTorch | 2.6.0+cu124 (manually installed) |
| numpy | >= 1.26.4, < 2.0 |

## Run Order

1. Steps 1-3: Cache redirect, install dependencies, build C++ extensions (first time only)
2. Restart kernel
3. Steps 4-10: Setup, background removal, shape generation, texture, export
