"""
StyleForge CUDA Kernels Package
Custom CUDA kernels for accelerated neural style transfer.

For ZeroGPU/HuggingFace: Pre-compiled kernels are downloaded from HF dataset.
For local: Kernels are JIT-compiled if prebuilt not available.
"""

import torch
import os
from pathlib import Path

# Try to import CUDA kernels, fall back gracefully
_CUDA_KERNELS_AVAILABLE = False
_FusedInstanceNorm2d = None
_KERNELS_COMPILED = False
_LOADED_KERNEL_FUNC = None

# Check if running on ZeroGPU
_ZERO_GPU = os.environ.get('SPACE_ID', '').startswith('hf.co') or os.environ.get('ZERO_GPU') == '1'

# Path to pre-compiled kernels
_PREBUILT_PATH = Path(__file__).parent / "prebuilt"
_PREBUILT_PATH.mkdir(exist_ok=True)

# HuggingFace dataset for prebuilt kernels
_KERNEL_DATASET = "oliau/styleforge-kernels"  # You'll need to create this dataset


def _download_kernels_from_dataset():
    """Download pre-compiled kernels from HuggingFace dataset."""
    try:
        from huggingface_hub import hf_hub_download
        import sys

        print(f"Looking for kernels in dataset: {_KERNEL_DATASET}")

        # Known kernel file name
        kernel_file = "fused_instance_norm.so"

        # Download directly to the kernels directory
        try:
            local_path = hf_hub_download(
                repo_id=_KERNEL_DATASET,
                filename=kernel_file,
                repo_type="dataset",
                local_dir=str(_PREBUILT_PATH.parent),
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded kernel: {kernel_file} -> {local_path}")
            return True
        except Exception as e:
            print(f"Failed to download {kernel_file}: {e}")
            # Try alternative paths in case the file is in a subdirectory
            for subdir in ["", "kernels/", "prebuilt/", "build/"]:
                try:
                    alt_path = subdir + kernel_file
                    local_path = hf_hub_download(
                        repo_id=_KERNEL_DATASET,
                        filename=alt_path,
                        repo_type="dataset",
                        local_dir=str(_PREBUILT_PATH.parent),
                        local_dir_use_symlinks=False
                    )
                    print(f"Successfully downloaded kernel from {alt_path}: {local_path}")
                    return True
                except Exception:
                    continue
            return False

    except ImportError as e:
        print(f"huggingface_hub not available: {e}")
        return False
    except Exception as e:
        print(f"Failed to download kernels from dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cuda_kernels():
    """Check if CUDA kernels are available."""
    return _CUDA_KERNELS_AVAILABLE


def get_fused_instance_norm(num_features, **kwargs):
    """
    Get FusedInstanceNorm2d module or PyTorch fallback.

    On ZeroGPU: Uses pre-compiled kernels if available.
    On local: May use custom fused kernels (prebuilt or JIT).
    """
    if _FusedInstanceNorm2d is not None:
        try:
            return _FusedInstanceNorm2d(num_features, **kwargs)
        except Exception:
            pass
    # Fallback to PyTorch (still GPU-accelerated, just not custom fused)
    return torch.nn.InstanceNorm2d(num_features, affine=kwargs.get('affine', True))


def load_prebuilt_kernels():
    """
    Try to load pre-compiled CUDA kernels from the kernels directory.
    On HuggingFace, downloads from dataset if local files not found.

    Returns True if successful, False otherwise.
    """
    global _FusedInstanceNorm2d, _CUDA_KERNELS_AVAILABLE, _KERNELS_COMPILED

    if _KERNELS_COMPILED:
        return _CUDA_KERNELS_AVAILABLE

    # Check for kernels in the kernels directory (parent of prebuilt) and prebuilt/
    kernels_dir = Path(__file__).parent
    kernel_files = list(kernels_dir.glob("*.so")) + list(kernels_dir.glob("*.pyd"))
    kernel_files += list(_PREBUILT_PATH.glob("*.so")) + list(_PREBUILT_PATH.glob("*.pyd"))

    # On HuggingFace Spaces, try downloading from dataset if not found locally
    if not kernel_files and _ZERO_GPU:
        print("No local pre-compiled kernels found. Trying HuggingFace dataset...")
        if _download_kernels_from_dataset():
            # Check again after download - look in kernels directory
            kernel_files = list(kernels_dir.glob("*.so")) + list(kernels_dir.glob("*.pyd"))
            kernel_files += list(_PREBUILT_PATH.glob("*.so")) + list(_PREBUILT_PATH.glob("*.pyd"))

    if not kernel_files:
        print("No pre-compiled kernels found")
        return False

    print(f"Found kernel files: {[f.name for f in kernel_files]}")

    try:
        import sys
        import ctypes

        # Try to load each kernel file
        for kernel_file in kernel_files:
            try:
                # First try to load as a Python extension module
                module_name = kernel_file.stem
                spec = __import__('importlib.util').util.spec_from_file_location(module_name, kernel_file)
                if spec and spec.loader:
                    mod = __import__('importlib.util').util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    print(f"Loaded pre-compiled kernel module: {kernel_file.name}")

                    # Check what functions are available in the module
                    available_funcs = [attr for attr in dir(mod) if not attr.startswith('_')]
                    print(f"Available functions in kernel: {available_funcs}")

                    # Try to find the forward function with common naming patterns
                    forward_func = None
                    for func_name in ['fused_instance_norm_forward', 'forward', 'fused_instance_norm',
                                      'instance_norm_forward', 'fused_inst_norm']:
                        if hasattr(mod, func_name):
                            forward_func = getattr(mod, func_name)
                            print(f"Using function: {func_name}")
                            break

                    if forward_func is None:
                        print(f"Warning: No suitable forward function found in {kernel_file.name}")
                        continue

                    # Store the kernel function globally for use with FusedInstanceNorm2d
                    _LOADED_KERNEL_FUNC = forward_func

                    # Create factory function that uses the wrapper with pre-loaded kernel
                    def make_fused_instance_norm(num_features, **kwargs):
                        from .instance_norm_wrapper import FusedInstanceNorm2d
                        # Pass the pre-loaded kernel function
                        return FusedInstanceNorm2d(num_features, kernel_func=forward_func, **kwargs)

                    _FusedInstanceNorm2d = make_fused_instance_norm
                    _CUDA_KERNELS_AVAILABLE = True
                    _KERNELS_COMPILED = True
                    print(f"Successfully initialized FusedInstanceNorm2d from {kernel_file.name}")
                    return True

            except Exception as e:
                print(f"Failed to load {kernel_file.name} as Python module: {e}")
                # Try loading as raw ctypes library
                try:
                    lib = ctypes.CDLL(str(kernel_file))
                    print(f"Loaded {kernel_file.name} as ctypes library")
                    # Could add ctypes wrapper here if needed
                except Exception as e2:
                    print(f"Failed to load {kernel_file.name} as ctypes: {e2}")
                continue

    except Exception as e:
        print(f"Failed to load prebuilt kernels: {e}")

    return False


def compile_kernels():
    """
    Compile CUDA kernels on-demand.

    On ZeroGPU: Downloads pre-compiled kernels from dataset.
    On local: Compiles custom CUDA kernels.
    """
    global _CUDA_KERNELS_AVAILABLE, _FusedInstanceNorm2d, _KERNELS_COMPILED

    if _KERNELS_COMPILED:
        return _CUDA_KERNELS_AVAILABLE

    # On ZeroGPU, try to download pre-compiled kernels from dataset
    if _ZERO_GPU:
        print("ZeroGPU mode: Attempting to download pre-compiled kernels from dataset...")
        if load_prebuilt_kernels():
            print("Successfully loaded pre-compiled CUDA kernels from dataset!")
            return True
        else:
            print("No pre-compiled kernels found in dataset, using PyTorch GPU fallback")
            _KERNELS_COMPILED = True
            return False

    # First, try pre-compiled kernels (for local too)
    if load_prebuilt_kernels():
        print("Using pre-compiled CUDA kernels!")
        return True

    if not torch.cuda.is_available():
        _KERNELS_COMPILED = True
        return False

    try:
        from .instance_norm_wrapper import FusedInstanceNorm2d
        _FusedInstanceNorm2d = FusedInstanceNorm2d
        _CUDA_KERNELS_AVAILABLE = True
        _KERNELS_COMPILED = True
        print("CUDA kernels compiled successfully!")
        return True
    except Exception as e:
        print(f"Failed to compile CUDA kernels: {e}")
        print("Using PyTorch InstanceNorm2d fallback")
        _KERNELS_COMPILED = True
        return False


# Auto-compile on import for non-ZeroGPU environments with CUDA
if _ZERO_GPU:
    # On ZeroGPU, try to download pre-compiled kernels
    print("ZeroGPU detected: Attempting to load pre-compiled kernels from dataset...")
    if load_prebuilt_kernels():
        print("Using pre-compiled CUDA kernels from dataset!")
    else:
        print("No pre-compiled kernels available, using PyTorch GPU fallback")
    _KERNELS_COMPILED = True
elif torch.cuda.is_available():
    compile_kernels()


__all__ = [
    'check_cuda_kernels',
    'get_fused_instance_norm',
    'FusedInstanceNorm2d',
    'compile_kernels',
    'load_prebuilt_kernels',
]
