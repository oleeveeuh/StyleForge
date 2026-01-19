"""
StyleForge - Fused Conv2d + InstanceNorm2d + ReLU Wrapper

Python interface for the fused convolution kernel.

Fuses: Conv2d → InstanceNorm2d → ReLU

This is a critical optimization for style transfer networks where
Conv+InstanceNorm+ReLU appears 15-20 times per forward pass.

Performance Target: 5-8x speedup over PyTorch sequential for small feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Union

from utils import compile_inline

# Global module cache
_conv_fusion_module = None


def get_conv_fusion_module():
    """Lazy-load and compile the conv fusion kernel."""
    global _conv_fusion_module

    if _conv_fusion_module is not None:
        return _conv_fusion_module

    kernel_path = Path(__file__).parent / "conv_fusion.cu"

    if not kernel_path.exists():
        raise FileNotFoundError(f"Conv fusion kernel not found at {kernel_path}")

    cuda_source = kernel_path.read_text()

    print("Compiling fused Conv+InstanceNorm+ReLU kernel...")
    _conv_fusion_module = compile_inline(
        name='conv_fusion',
        cuda_source=cuda_source,
        functions=['fused_conv_instance_norm_relu'],
        build_directory=Path('build'),
        verbose=False
    )
    print("Conv fusion compilation complete!")

    return _conv_fusion_module


class FusedConvInstanceNormReLU(nn.Module):
    """
    Fused Convolution + Instance Normalization + ReLU Module

    Replaces the common pattern:
        nn.Conv2d → nn.InstanceNorm2d → nn.ReLU

    With a single fused kernel for 5-8x speedup on small feature maps.

    This is particularly useful for:
        - Style transfer networks (Johnson et al.)
        - Residual blocks in generative models
        - Any architecture with repeated Conv-IN-ReLU patterns

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (1, 3, 4, or 5)
        stride: Convolution stride (default: 1)
        padding: Convolution padding (default: 1 for kernel_size=3)
        eps: Epsilon for instance norm numerical stability
        bias: Use bias in convolution (default: True)
        affine: Use affine transform in instance norm (default: True)

    Example:
        >>> # Standard residual block pattern
        >>> block = nn.Sequential(
        ...     FusedConvInstanceNormReLU(64, 64, kernel_size=3),
        ...     FusedConvInstanceNormReLU(64, 64, kernel_size=3),
        ... )
        >>> x = torch.randn(1, 64, 256, 256).cuda()
        >>> y = block(x)
        >>> print(y.shape)  # [1, 64, 256, 256]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        eps: float = 1e-5,
        bias: bool = True,
        affine: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps

        # Default padding based on kernel size
        if padding is None:
            if kernel_size == 1:
                padding = 0
            elif kernel_size == 3:
                padding = 1
            elif kernel_size == 4:
                padding = 1
            elif kernel_size == 5:
                padding = 2
            else:
                raise ValueError(f"Unsupported kernel size: {kernel_size}")

        self.padding = padding
        self.affine = affine

        # Convolution parameters
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # InstanceNorm parameters (affine transform)
        if affine:
            self.gamma = nn.Parameter(torch.ones(out_channels))
            self.beta = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_buffer('gamma', torch.ones(out_channels))
            self.register_buffer('beta', torch.zeros(out_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        # Kaiming initialization for conv weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # InstanceNorm parameters are already initialized to ones/zeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused Conv+InstanceNorm+ReLU kernel.

        Args:
            x: Input tensor [N, C_in, H, W]

        Returns:
            Output tensor [N, C_out, H_out, W_out]
        """
        module = get_conv_fusion_module()

        # Prepare bias tensor
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)

        with torch.cuda.nvtx.range("fused_conv_in_relu"):
            output = module.fused_conv_instance_norm_relu(
                x.contiguous(),
                self.weight.contiguous(),
                bias.contiguous(),
                self.gamma.contiguous(),
                self.beta.contiguous(),
                self.stride,
                self.padding,
                self.eps
            )

        return output

    def load_from_pytorch(
        self,
        conv: nn.Conv2d,
        instance_norm: nn.InstanceNorm2d
    ):
        """
        Load weights from existing PyTorch layers.

        Useful for converting pretrained models.

        Args:
            conv: nn.Conv2d layer
            instance_norm: nn.InstanceNorm2d layer
        """
        # Copy conv weights
        self.weight.data.copy_(conv.weight.data)
        if conv.bias is not None and self.bias is not None:
            self.bias.data.copy_(conv.bias.data)

        # Copy instance norm parameters
        if hasattr(instance_norm, 'weight') and instance_norm.weight is not None:
            self.gamma.data.copy_(instance_norm.weight.data)
        if hasattr(instance_norm, 'bias') and instance_norm.bias is not None:
            self.beta.data.copy_(instance_norm.bias.data)

    def extra_repr(self) -> str:
        return (f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, '
                f'stride={self.stride}, '
                f'padding={self.padding}')


class ResidualBlock(nn.Module):
    """
    Residual block using fused Conv+InstanceNorm+ReLU.

    Standard architecture in style transfer networks:
        Input → Conv → IN → ReLU → Conv → IN → + Input → ReLU

    Args:
        channels: Number of input/output channels
        kernel_size: Convolution kernel size (default: 3)
        stride: Convolution stride (default: 1)

    Example:
        >>> block = ResidualBlock(64).cuda()
        >>> x = torch.randn(1, 64, 128, 128).cuda()
        >>> y = block(x)
        >>> print(y.shape)  # [1, 64, 128, 128]
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1
    ):
        super().__init__()

        self.conv1 = FusedConvInstanceNormReLU(
            channels, channels, kernel_size, stride
        )
        self.conv2 = FusedConvInstanceNormReLU(
            channels, channels, kernel_size, stride
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

    def load_from_pytorch_block(
        self,
        conv1: nn.Conv2d,
        in1: nn.InstanceNorm2d,
        relu1: nn.ReLU,
        conv2: nn.Conv2d,
        in2: nn.InstanceNorm2d,
        relu2: nn.ReLU
    ):
        """Load weights from a PyTorch residual block."""
        self.conv1.load_from_pytorch(conv1, in1)
        self.conv2.load_from_pytorch(conv2, in2)


def benchmark_conv_fusion_vs_pytorch(
    batch_size: int = 1,
    in_channels: int = 64,
    out_channels: int = 64,
    height: int = 128,
    width: int = 128,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    iterations: int = 100
):
    """
    Benchmark fused Conv+InstanceNorm+ReLU against PyTorch sequential.

    Args:
        batch_size: Batch size
        in_channels: Input channels
        out_channels: Output channels
        height: Input height
        width: Input width
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results
    """
    import numpy as np

    print(f"\n{'='*70}")
    print(f"Fused Conv+InstanceNorm+ReLU Benchmark")
    print(f"{'='*70}")
    print(f"Config: [{batch_size}, {in_channels}, {height}, {width}] → "
          f"[{batch_size}, {out_channels}, {height}, {width}]")
    print(f"Kernel: {kernel_size}x{kernel_size}, stride={stride}, padding={padding}")

    x = torch.randn(batch_size, in_channels, height, width, device='cuda')

    results = {}

    # ============================================================
    # PyTorch Baseline (3 separate operations)
    # ============================================================
    print("\n1. PyTorch Sequential (Conv2d → InstanceNorm2d → ReLU)...")

    conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                     stride=stride, padding=padding, bias=True).cuda().eval()
    instance_norm = nn.InstanceNorm2d(out_channels, affine=True).cuda().eval()
    relu = nn.ReLU(inplace=False).cuda()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            out = conv(x)
            out = instance_norm(out)
            out = relu(out)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            out = conv(x)
            out = instance_norm(out)
            out = relu(out)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    pytorch_out = out.clone()
    results['pytorch'] = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'name': 'PyTorch Sequential'
    }
    print(f"   {results['pytorch']['mean_ms']:.3f} ± {results['pytorch']['std_ms']:.3f} ms")

    # ============================================================
    # Fused Conv+InstanceNorm+ReLU
    # ============================================================
    print("\n2. Fused Conv+InstanceNorm+ReLU Kernel...")

    try:
        fused = FusedConvInstanceNormReLU(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        ).cuda().eval()

        # Copy weights from PyTorch layers for fair comparison
        with torch.no_grad():
            fused.weight.copy_(conv.weight)
            if conv.bias is not None:
                fused.bias.copy_(conv.bias)
            fused.gamma.copy_(instance_norm.weight)
            fused.beta.copy_(instance_norm.bias)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                out = fused(x)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                out = fused(x)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        fused_out = out.clone()
        results['fused'] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'name': 'Fused Conv+IN+ReLU'
        }
        print(f"   {results['fused']['mean_ms']:.3f} ± {results['fused']['std_ms']:.3f} ms")

        # ============================================================
        # Correctness Check
        # ============================================================
        print("\n3. Correctness Check...")
        max_diff = torch.max(torch.abs(pytorch_out - fused_out)).item()
        mean_diff = torch.mean(torch.abs(pytorch_out - fused_out)).item()

        print(f"   Max difference:  {max_diff:.2e}")
        print(f"   Mean difference: {mean_diff:.2e}")

        if max_diff < 1e-4:
            print("   ✓ Outputs match (tolerance: 1e-4)")
        elif max_diff < 1e-3:
            print("   ⚠ Outputs mostly match (tolerance: 1e-3)")
        else:
            print("   ✗ Outputs differ significantly!")

        # ============================================================
        # Summary
        # ============================================================
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        baseline = results['pytorch']['mean_ms']
        fused_time = results['fused']['mean_ms']
        speedup = baseline / fused_time

        print(f"\nPyTorch:  {baseline:.3f} ms")
        print(f"Fused:    {fused_time:.3f} ms")
        print(f"\nSpeedup:  {speedup:.2f}x")

        if speedup < 1.0:
            print("⚠️  CUDA slower - check implementation")
        elif speedup < 2.0:
            print("✓ Modest speedup")
        elif speedup < 5.0:
            print("✓✓ Good speedup")
        else:
            print("✓✓✓ Excellent speedup!")

    except Exception as e:
        print(f"   ❌ CUDA kernel failed: {e}")
        import traceback
        traceback.print_exc()
        results['fused'] = None

    return results


def run_comprehensive_benchmark():
    """Run benchmarks across different configurations."""

    print("\n" + "="*70)
    print("Comprehensive Conv+InstanceNorm+ReLU Fusion Benchmark")
    print("="*70)

    configs = [
        # (name, batch, in_ch, out_ch, h, w, kernel_size)
        ("Small feature map", 1, 64, 64, 64, 64, 3),
        ("Medium feature map", 1, 128, 128, 128, 128, 3),
        ("Large feature map", 1, 64, 64, 256, 256, 3),
        ("Residual block size", 1, 128, 128, 32, 32, 3),
        ("1x1 conv (bottleneck)", 1, 256, 64, 64, 64, 1),
        ("Downsample block", 1, 64, 128, 128, 128, 3),
    ]

    all_results = {}

    for name, batch, in_ch, out_ch, h, w, k in configs:
        stride = 2 if "Downsample" in name else 1
        padding = 1

        results = benchmark_conv_fusion_vs_pytorch(
            batch_size=batch,
            in_channels=in_ch,
            out_channels=out_ch,
            height=h,
            width=w,
            kernel_size=k,
            stride=stride,
            padding=padding,
            iterations=100
        )

        all_results[name] = results

    # Final summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)

    for name, results in all_results.items():
        if results.get('fused') is not None:
            baseline = results['pytorch']['mean_ms']
            fused_time = results['fused']['mean_ms']
            speedup = baseline / fused_time
            print(f"{name:25s}: {speedup:.2f}x speedup")

    return all_results


if __name__ == "__main__":
    # Run benchmark if executed directly
    run_comprehensive_benchmark()
