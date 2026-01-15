"""
StyleForge - Fused Instance Normalization Wrapper

Python interface for the fused InstanceNorm CUDA kernel.

Fuses: Mean → Variance → Normalize → Affine Transform

Performance Target: 3-5x speedup over PyTorch nn.InstanceNorm2d
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from ..utils import compile_inline

# Global module cache
_instance_norm_module = None


def get_instance_norm_module():
    """Lazy-load and compile the InstanceNorm kernel."""
    global _instance_norm_module

    if _instance_norm_module is not None:
        return _instance_norm_module

    kernel_path = Path(__file__).parent / "instance_norm.cu"

    if not kernel_path.exists():
        raise FileNotFoundError(f"InstanceNorm kernel not found at {kernel_path}")

    cuda_source = kernel_path.read_text()

    print("Compiling fused InstanceNorm kernel...")
    _instance_norm_module = compile_inline(
        name='fused_instance_norm',
        cuda_source=cuda_source,
        functions=['forward'],
        build_directory=Path('build'),
        verbose=False
    )
    print("InstanceNorm compilation complete!")

    return _instance_norm_module


class FusedInstanceNorm2d(nn.Module):
    """
    Fused Instance Normalization 2D Module

    Fuses the entire InstanceNorm operation into a single kernel:
        Compute Mean → Compute Variance → Normalize → Affine Transform

    Args:
        num_features: Number of channels (C)
        eps: Small value for numerical stability
        affine: Use learnable affine parameters (gamma, beta)
        track_running_stats: Not supported (always False for inference)
        use_vectorized: Use float4 vectorization (default: True)

    Example:
        >>> norm = FusedInstanceNorm2d(64).cuda()
        >>> x = torch.randn(2, 64, 128, 128).cuda()
        >>> y = norm(x)
        >>> print(y.shape)  # [2, 64, 128, 128]
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
        self.track_running_stats = False  # Always False for fused version

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer('gamma', torch.ones(num_features))
            self.register_buffer('beta', torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused InstanceNorm kernel.

        Args:
            x: Input tensor [batch, channels, height, width]

        Returns:
            Output tensor [batch, channels, height, width]
        """
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D (B, C, H, W), got {x.dim()}D")

        module = get_instance_norm_module()

        with torch.cuda.nvtx.range("fused_instance_norm_forward"):
            output = module.forward(
                x.contiguous(),
                self.gamma,
                self.beta,
                self.eps,
                self.use_vectorized
            )

        return output

    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, eps={self.eps}, affine={self.gamma.requires_grad}'


def benchmark_instance_norm_vs_pytorch(
    batch_size: int = 2,
    channels: int = 64,
    height: int = 128,
    width: int = 128,
    iterations: int = 100
):
    """
    Benchmark fused InstanceNorm against PyTorch.

    Returns:
        Dictionary with benchmark results
    """
    import numpy as np

    print(f"\nBenchmarking InstanceNorm ({batch_size}x{channels}x{height}x{width})...")
    print("=" * 70)

    x = torch.randn(batch_size, channels, height, width, device='cuda')

    results = {}

    # ----------------------------------------
    # PyTorch Baseline
    # ----------------------------------------
    print("\n1. PyTorch InstanceNorm2d...")

    norm_pytorch = nn.InstanceNorm2d(channels, affine=True).cuda().eval()

    times = []
    for _ in range(10):
        with torch.no_grad():
            _ = norm_pytorch(x)

    torch.cuda.synchronize()
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = norm_pytorch(x)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    results['pytorch'] = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'name': 'PyTorch InstanceNorm2d'
    }
    print(f"   {results['pytorch']['mean_ms']:.2f} ± {results['pytorch']['std_ms']:.2f} ms")

    # ----------------------------------------
    # Fused InstanceNorm
    # ----------------------------------------
    print("\n2. Fused InstanceNorm...")

    norm_fused = FusedInstanceNorm2d(channels, use_vectorized=True).cuda().eval()

    # Copy weights for fair comparison
    with torch.no_grad():
        norm_fused.gamma.copy_(norm_pytorch.weight)
        norm_fused.beta.copy_(norm_pytorch.bias)

    times = []
    for _ in range(10):
        with torch.no_grad():
            _ = norm_fused(x)

    torch.cuda.synchronize()
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = norm_fused(x)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    results['fused'] = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'name': 'Fused InstanceNorm'
    }
    print(f"   {results['fused']['mean_ms']:.2f} ± {results['fused']['std_ms']:.2f} ms")

    # ----------------------------------------
    # Summary
    # ----------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline = results['pytorch']['mean_ms']
    fused_time = results['fused']['mean_ms']

    print(f"\nPyTorch:  {baseline:.2f} ms")
    print(f"Fused:    {fused_time:.2f} ms")
    print(f"\n🚀 Fused InstanceNorm is {baseline/fused_time:.2f}x faster than PyTorch!")

    return results


if __name__ == "__main__":
    # Run benchmark if executed directly
    benchmark_instance_norm_vs_pytorch()
