# Pre-Compiled CUDA Kernels

This directory contains pre-compiled CUDA kernels for use on Hugging Face Spaces.

## How to Compile Kernels Locally

To compile the CUDA kernels locally and upload them here:

### 1. Compile Locally

Run this script from the `huggingface-space` directory:

```bash
python compile_kernels.py
```

Or compile manually:

```bash
cd huggingface-space
python -c "
from kernels.cuda_build import compile_inline
from pathlib import Path

cuda_source = (Path('kernels') / 'instance_norm.cu').read_text()
module = compile_inline(
    name='fused_instance_norm',
    cuda_source=cuda_source,
    functions=['forward'],
    build_directory=Path('build'),
    verbose=True
)
print('Compiled successfully!')
print(f'Module location: {module.__file__}')
"
```

### 2. Copy Compiled File

After compilation, copy the compiled `.so` file to this directory:

```bash
# Find the compiled file (usually in build/)
find build/ -name "*.so" -exec cp {} kernels/prebuilt/ \;
```

### 3. Commit and Push

```bash
git add kernels/prebuilt/
git commit -m "Add pre-compiled CUDA kernels"
git push
```

## Notes

- The compiled kernels are architecture-specific (e.g., `sm_70`, `sm_75`, `sm_86`)
- Hugging Face Spaces typically use Tesla T4 (sm_75) or A100 (sm_80)
- For maximum compatibility, compile with multiple compute capabilities

## Current Status

No pre-compiled kernels found. The app will use PyTorch's InstanceNorm2d fallback,
which is still GPU-accelerated but not as fast as custom fused kernels.
