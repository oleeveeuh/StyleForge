"""
StyleForge - Fused Instance Normalization Wrapper
Python interface for the fused InstanceNorm CUDA kernel.

On ZeroGPU: Uses pre-compiled kernels from HuggingFace dataset.
On local: JIT compiles from source.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import os

# Check if running on ZeroGPU
_ZERO_GPU = os.environ.get('SPACE_ID', '').startswith('hf.co') or os.environ.get('ZERO_GPU') == '1'

# Import local build utilities (only if not on ZeroGPU)
if not _ZERO_GPU:
    from .cuda_build import compile_inline

# Global module cache
_instance_norm_module = None
_cuda_available = None


def check_cuda_available():
    """Check if CUDA is available and kernels can be compiled."""
    global _cuda_available
    if _cuda_available is not None:
        return _cuda_available

    _cuda_available = torch.cuda.is_available()
    return _cuda_available


def get_instance_norm_module():
    """Lazy-load and compile the InstanceNorm kernel."""
    global _instance_norm_module

    if _instance_norm_module is not None:
        return _instance_norm_module

    # On ZeroGPU, pre-compiled kernels should be loaded by __init__.py
    # This function is only for local JIT compilation
    if _ZERO_GPU:
        raise RuntimeError("ZeroGPU mode: Pre-compiled kernels should be loaded via __init__.py")

    if not check_cuda_available():
        raise RuntimeError("CUDA is not available. Cannot use fused InstanceNorm kernel.")

    kernel_path = Path(__file__).parent / "instance_norm.cu"

    if not kernel_path.exists():
        raise FileNotFoundError(f"InstanceNorm kernel not found at {kernel_path}")

    cuda_source = kernel_path.read_text()

    print("Compiling fused InstanceNorm kernel...")
    try:
        _instance_norm_module = compile_inline(
            name='fused_instance_norm',
            cuda_source=cuda_source,
            functions=['forward'],
            build_directory=Path('build'),
            verbose=False
        )
        print("InstanceNorm compilation complete!")
    except Exception as e:
        print(f"Failed to compile InstanceNorm kernel: {e}")
        print("Falling back to PyTorch implementation.")
        raise

    return _instance_norm_module


class FusedInstanceNorm2d(nn.Module):
    """
    Fused Instance Normalization 2D Module with automatic fallback.

    On ZeroGPU: Uses pre-compiled kernels if available.
    On local: May use JIT-compiled kernels.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = False,
        use_vectorized: bool = True,
        kernel_func: Optional[callable] = None  # Pre-loaded kernel function
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.use_vectorized = use_vectorized
        self.track_running_stats = False
        self._kernel_func = kernel_func  # Pre-loaded from __init__.py

        # Enable CUDA if kernel function is provided OR not on ZeroGPU with CUDA available
        self._use_cuda = (self._kernel_func is not None) or (check_cuda_available() if not _ZERO_GPU else False)

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer('gamma', torch.ones(num_features))
            self.register_buffer('beta', torch.zeros(num_features))

        # Fallback to PyTorch InstanceNorm
        self._pytorch_norm = nn.InstanceNorm2d(num_features, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D (B, C, H, W), got {x.dim()}D")

        # Use pre-compiled kernel if available
        if self._kernel_func is not None and x.is_cuda:
            try:
                result = self._kernel_func(
                    x.contiguous(),
                    self.gamma,
                    self.beta,
                    self.eps
                )
                return result
            except Exception as e:
                print(f"Custom kernel failed: {e}, falling back to PyTorch")
                # Continue to PyTorch fallback

        # Use CUDA kernel if available and on CUDA device (local JIT compilation)
        if self._use_cuda and x.is_cuda and not _ZERO_GPU and self._kernel_func is None:
            try:
                module = get_instance_norm_module()
                output = module.forward(
                    x.contiguous(),
                    self.gamma,
                    self.beta,
                    self.eps,
                    self.use_vectorized
                )
                return output
            except Exception:
                # Fallback to PyTorch
                pass

        # PyTorch fallback (still GPU accelerated, just not custom fused)
        return self._pytorch_norm(x)


# Alias for compatibility
FusedInstanceNorm2dAuto = FusedInstanceNorm2d
