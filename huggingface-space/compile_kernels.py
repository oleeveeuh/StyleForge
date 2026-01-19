#!/usr/bin/env python3
"""
Compile CUDA kernels locally for deployment to Hugging Face Spaces.
"""

import sys
import os
import torch
from pathlib import Path

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("StyleForge CUDA Kernel Compiler")
print("=" * 60)
print()

# Check CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available on this system.")
    print("This script requires a CUDA-capable GPU.")
    sys.exit(1)

print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Get compute capability
major, minor = torch.cuda.get_device_capability(0)
compute_capability = f"{major}.{minor}"
print(f"Compute Capability: {compute_capability}")
print()

# Create prebuilt directory
prebuilt_dir = Path("kernels/prebuilt")
prebuilt_dir.mkdir(exist_ok=True, parents=True)

print("Compiling CUDA kernels...")
print("-" * 60)

try:
    # Import PyTorch CUDA extension utilities
    from torch.utils.cpp_extension import load_inline, CUDA_HOME

    if CUDA_HOME is None:
        print("ERROR: CUDA_HOME is not set. CUDA toolkit may not be installed.")
        sys.exit(1)

    print(f"CUDA Home: {CUDA_HOME}")

    # Read CUDA source
    kernel_path = Path("kernels/instance_norm.cu")
    if not kernel_path.exists():
        print(f"ERROR: Kernel source not found at {kernel_path}")
        sys.exit(1)

    cuda_source = kernel_path.read_text()
    print(f"Loaded CUDA source: {len(cuda_source)} bytes")

    # Architecture-specific flags for Hugging Face GPUs
    extra_cuda_cflags = ['-O3', '--use_fast_math']
    hf_arch_flags = [
        '-gencode=arch=compute_70,code=sm_70',  # V100
        '-gencode=arch=compute_75,code=sm_75',  # T4
        '-gencode=arch=compute_80,code=sm_80',  # A100
    ]
    extra_cuda_cflags.extend(hf_arch_flags)

    print("Build flags:", ' '.join(extra_cuda_cflags))
    print()
    print("Compiling... (this may take 1-2 minutes)")

    # Compile the kernel
    # Note: PyTorch 2.x requires cpp_sources even if empty (bindings are in CUDA)
    module = load_inline(
        name='fused_instance_norm',
        cpp_sources=[],  # Empty since bindings are in the .cu file
        cuda_sources=[cuda_source],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False
    )

    print()
    print("-" * 60)
    print("Compilation successful!")
    print()

    # Find the compiled library
    import torch.utils.cpp_extension
    build_dir = Path(torch.utils.cpp_extension._get_build_directory('fused_instance_norm', False))
    print(f"Build directory: {build_dir}")

    so_files = list(build_dir.rglob("*.so")) + list(build_dir.rglob("*.pyd"))

    if not so_files:
        print("ERROR: No compiled .so/.pyd file found")
        sys.exit(1)

    # Copy to prebuilt directory
    import shutil
    for src_file in so_files:
        dst_file = prebuilt_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        size_kb = dst_file.stat().st_size / 1024
        print(f"Copied: {dst_file.name} ({size_kb:.1f} KB)")

    print()
    print("=" * 60)
    print("Kernel compilation complete!")
    print(f"Pre-compiled kernels saved to: {prebuilt_dir}")
    print()
    print("Download the .so file and add it to your local repo:")
    print("  kernels/prebuilt/" + list(prebuilt_dir.glob("*.so"))[0].name if list(prebuilt_dir.glob("*.so")) else "")
    print("=" * 60)

except Exception as e:
    print()
    print("-" * 60)
    print("ERROR: Compilation failed!")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
