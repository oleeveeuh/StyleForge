"""
StyleForge - Fused Feed-Forward Network Wrapper

Python interface for the fused FFN CUDA kernel.

Fuses: Linear → GELU → Linear → Bias → Residual

Performance Target: 4-5x speedup over PyTorch sequential
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from ..utils import compile_inline

# Global module cache
_ffn_module = None


def get_ffn_module():
    """Lazy-load and compile the FFN kernel."""
    global _ffn_module

    if _ffn_module is not None:
        return _ffn_module

    kernel_path = Path(__file__).parent / "ffn.cu"

    if not kernel_path.exists():
        raise FileNotFoundError(f"FFN kernel not found at {kernel_path}")

    cuda_source = kernel_path.read_text()

    print("Compiling fused FFN kernel...")
    _ffn_module = compile_inline(
        name='fused_ffn',
        cuda_source=cuda_source,
        functions=['forward'],
        build_directory=Path('build'),
        verbose=False
    )
    print("FFN compilation complete!")

    return _ffn_module


class FusedFFN(nn.Module):
    """
    Fused Feed-Forward Network Module

    Fuses the entire FFN block into a single kernel:
        Linear(embed_dim, ffn_dim) → GELU → Linear(ffn_dim, embed_dim) + Residual

    Args:
        embed_dim: Input/output embedding dimension
        ffn_dim: Hidden dimension of FFN (typically 4x embed_dim)
        dropout: Dropout probability (not used in V1)
        bias: Use bias in linear layers

    Example:
        >>> ffn = FusedFFN(embed_dim=128, ffn_dim=512).cuda()
        >>> x = torch.randn(2, 256, 128).cuda()
        >>> y = ffn(x)
        >>> print(y.shape)  # [2, 256, 128]
    """

    def __init__(
        self,
        embed_dim: int = 128,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim

        # FC1: embed_dim → ffn_dim
        self.fc1_weight = nn.Parameter(torch.empty(embed_dim, ffn_dim))
        self.fc1_bias = nn.Parameter(torch.empty(ffn_dim)) if bias else None

        # FC2: ffn_dim → embed_dim
        self.fc2_weight = nn.Parameter(torch.empty(ffn_dim, embed_dim))
        self.fc2_bias = nn.Parameter(torch.empty(embed_dim)) if bias else None

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform"""
        nn.init.xavier_uniform_(self.fc1_weight)
        nn.init.xavier_uniform_(self.fc2_weight)

        if self.fc1_bias is not None:
            nn.init.zeros_(self.fc1_bias)
        if self.fc2_bias is not None:
            nn.init.zeros_(self.fc2_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused FFN kernel.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        module = get_ffn_module()

        # Transpose weights for kernel layout [out, in] → [in, out]
        w1_t = self.fc1_weight.T.contiguous()
        w2_t = self.fc2_weight.T.contiguous()

        # Create zero biases if not used
        b1 = self.fc1_bias if self.fc1_bias is not None else torch.zeros(
            self.ffn_dim, device=x.device
        )
        b2 = self.fc2_bias if self.fc2_bias is not None else torch.zeros(
            self.embed_dim, device=x.device
        )

        with torch.cuda.nvtx.range("fused_ffn_forward"):
            output = module.forward(
                x.contiguous(),
                w1_t,
                b1,
                w2_t,
                b2
            )

        # Apply dropout if training
        if self.training and self.dropout.p > 0:
            output = self.dropout(output)

        return output

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, ffn_dim={self.ffn_dim}'


def benchmark_ffn_vs_pytorch(
    batch_size: int = 2,
    seq_len: int = 256,
    embed_dim: int = 128,
    ffn_dim: int = 512,
    iterations: int = 100
):
    """
    Benchmark fused FFN against PyTorch sequential.

    Returns:
        Dictionary with benchmark results
    """
    import numpy as np

    print(f"\nBenchmarking FFN ({batch_size}x{seq_len}x{embed_dim})...")
    print("=" * 70)

    x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')

    results = {}

    # ----------------------------------------
    # PyTorch Baseline
    # ----------------------------------------
    print("\n1. PyTorch Sequential FFN...")

    ffn_pytorch = nn.Sequential(
        nn.Linear(embed_dim, ffn_dim),
        nn.GELU(),
        nn.Linear(ffn_dim, embed_dim)
    ).cuda().eval()

    times = []
    for _ in range(10):
        with torch.no_grad():
            _ = ffn_pytorch(x)

    torch.cuda.synchronize()
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = ffn_pytorch(x)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    results['pytorch'] = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'name': 'PyTorch Sequential'
    }
    print(f"   {results['pytorch']['mean_ms']:.2f} ± {results['pytorch']['std_ms']:.2f} ms")

    # ----------------------------------------
    # Fused FFN
    # ----------------------------------------
    print("\n2. Fused FFN Kernel...")

    ffn_fused = FusedFFN(embed_dim, ffn_dim).cuda().eval()

    times = []
    for _ in range(10):
        with torch.no_grad():
            _ = ffn_fused(x)

    torch.cuda.synchronize()
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = ffn_fused(x)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    results['fused'] = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'name': 'Fused FFN'
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
    print(f"\n🚀 Fused FFN is {baseline/fused_time:.2f}x faster than PyTorch!")

    return results


if __name__ == "__main__":
    # Run benchmark if executed directly
    benchmark_ffn_vs_pytorch()
