"""
StyleForge CUDA Kernels Package
Custom CUDA kernels for accelerated neural style transfer.

For ZeroGPU: Pre-compiled kernels are loaded from prebuilt/.
For local: Kernels are JIT-compiled if prebuilt not available.
"""

import torch
import os
from pathlib import Path

# Try to import CUDA kernels, fall back gracefully
_CUDA_KERNELS_AVAILABLE = False
_FusedInstanceNorm2d = None
_KERNELS_COMPILED = False

# Check if running on ZeroGPU
_ZERO_GPU = os.environ.get('SPACE_ID', '').startswith('hf.co') or os.environ.get('ZERO_GPU') == '1'

# Path to pre-compiled kernels
_PREBUILT_PATH = Path(__file__).parent / "prebuilt"


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
    Try to load pre-compiled CUDA kernels from prebuilt/ directory.

    Returns True if successful, False otherwise.
    """
    global _FusedInstanceNorm2d, _CUDA_KERNELS_AVAILABLE, _KERNELS_COMPILED

    if _KERNELS_COMPILED:
        return _CUDA_KERNELS_AVAILABLE

    # Check if prebuilt kernels exist
    prebuilt_files = list(_PREBUILT_PATH.glob("*.so")) + list(_PREBUILT_PATH.glob("*.pyd"))
    if not prebuilt_files:
        print("No pre-compiled kernels found in prebuilt/")
        return False

    try:
        # Try to import from prebuilt directory
        import sys
        if str(_PREBUILT_PATH) not in sys.path:
            sys.path.insert(0, str(_PREBUILT_PATH))

        # Try to load the prebuilt module
        for kernel_file in prebuilt_files:
            try:
                # Import the compiled module
                module_name = kernel_file.stem
                spec = __import__('importlib.util').util.spec_from_file_location(module_name, kernel_file)
                if spec and spec.loader:
                    mod = __import__('importlib.util').util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    print(f"Loaded pre-compiled kernel: {kernel_file.name}")

                    # Create FusedInstanceNorm2d class
                    class PrebuiltFusedInstanceNorm2d(torch.nn.Module):
                        def __init__(self, num_features, **kwargs):
                            super().__init__()
                            self.num_features = num_features
                            self.eps = kwargs.get('eps', 1e-5)
                            if kwargs.get('affine', True):
                                self.gamma = torch.nn.Parameter(torch.ones(num_features))
                                self.beta = torch.nn.Parameter(torch.zeros(num_features))
                            else:
                                self.register_buffer('gamma', torch.ones(num_features))
                                self.register_buffer('beta', torch.zeros(num_features))
                            self._pytorch_norm = torch.nn.InstanceNorm2d(num_features, **kwargs)

                        def forward(self, x):
                            try:
                                return mod.fused_instance_norm_forward(
                                    x.contiguous(), self.gamma, self.beta, self.eps
                                )
                            except Exception:
                                return self._pytorch_norm(x)

                    _FusedInstanceNorm2d = PrebuiltFusedInstanceNorm2d
                    _CUDA_KERNELS_AVAILABLE = True
                    _KERNELS_COMPILED = True
                    return True
            except Exception as e:
                print(f"Failed to load {kernel_file.name}: {e}")
                continue

    except Exception as e:
        print(f"Failed to load prebuilt kernels: {e}")

    return False


def compile_kernels():
    """
    Compile CUDA kernels on-demand.

    On ZeroGPU: Tries pre-compiled kernels first.
    On local: Compiles custom CUDA kernels.
    """
    global _CUDA_KERNELS_AVAILABLE, _FusedInstanceNorm2d, _KERNELS_COMPILED

    if _KERNELS_COMPILED:
        return _CUDA_KERNELS_AVAILABLE

    # First, try pre-compiled kernels
    if load_prebuilt_kernels():
        print("Using pre-compiled CUDA kernels!")
        return True

    # Fall back to JIT compilation (only on local, not ZeroGPU)
    if _ZERO_GPU:
        print("ZeroGPU mode: No pre-compiled kernels found, using PyTorch fallback")
        _KERNELS_COMPILED = True
        return False

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
if torch.cuda.is_available() and not _ZERO_GPU:
    compile_kernels()
elif _ZERO_GPU:
    # On ZeroGPU, try prebuilt kernels
    if load_prebuilt_kernels():
        print("ZeroGPU: Using pre-compiled CUDA kernels!")
    else:
        print("ZeroGPU: No pre-compiled kernels, using PyTorch GPU fallback")


__all__ = [
    'check_cuda_kernels',
    'get_fused_instance_norm',
    'FusedInstanceNorm2d',
    'compile_kernels',
    'load_prebuilt_kernels',
]
