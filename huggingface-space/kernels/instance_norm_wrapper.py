"""
StyleForge - Fused Instance Normalization Wrapper
Python interface for the fused InstanceNorm CUDA kernel.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# Import local build utilities
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
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = False,
        use_vectorized: bool = True
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.use_vectorized = use_vectorized
        self.track_running_stats = False
        self._use_cuda = check_cuda_available()

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

        # Use CUDA kernel if available and on CUDA device
        if self._use_cuda and x.is_cuda:
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

        # PyTorch fallback
        return self._pytorch_norm(x)


# Alias for compatibility
FusedInstanceNorm2dAuto = FusedInstanceNorm2d
