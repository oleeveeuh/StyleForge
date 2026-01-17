#!/usr/bin/env python3
"""
StyleForge - Comprehensive Conv Fusion Benchmark

Benchmark script for comparing fused Conv2d+InstanceNorm2d+ReLU
against PyTorch's sequential implementation.

This pattern is ubiquitous in style transfer networks, appearing
15-20 times per forward pass in typical architectures.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from kernels.conv_fusion_wrapper import (
    FusedConvInstanceNormReLU,
    ResidualBlock,
    benchmark_conv_fusion_vs_pytorch,
    run_comprehensive_benchmark
)


# ============================================================================
# Custom Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """Runs comprehensive benchmarks for conv fusion."""

    def __init__(self, warmup: int = 20, iterations: int = 100):
        self.warmup = warmup
        self.iterations = iterations
        self.results = []

    def benchmark_pytorch_sequential(
        self,
        x: torch.Tensor,
        conv: nn.Conv2d,
        instance_norm: nn.InstanceNorm2d,
        relu: nn.ReLU
    ) -> Dict[str, float]:
        """Benchmark PyTorch sequential implementation."""
        # Warmup
        for _ in range(self.warmup):
            with torch.no_grad():
                out = conv(x)
                out = instance_norm(out)
                out = relu(out)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(self.iterations):
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

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
        }

    def benchmark_fused(
        self,
        x: torch.Tensor,
        fused: FusedConvInstanceNormReLU
    ) -> Dict[str, float]:
        """Benchmark fused kernel implementation."""
        # Warmup
        for _ in range(self.warmup):
            with torch.no_grad():
                out = fused(x)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(self.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                out = fused(x)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
        }

    def verify_correctness(
        self,
        pytorch_out: torch.Tensor,
        fused_out: torch.Tensor,
        tolerance: float = 1e-4
    ) -> Dict[str, Any]:
        """Verify that outputs match within tolerance."""
        diff = torch.abs(pytorch_out - fused_out)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        return {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'passed': max_diff < tolerance,
            'tolerance': tolerance
        }

    def run_single_benchmark(
        self,
        name: str,
        batch_size: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ) -> Dict[str, Any]:
        """Run a single benchmark configuration."""
        print(f"\n{'='*70}")
        print(f"{name}: [{batch_size}, {in_channels}, {height}, {width}] → "
              f"[{batch_size}, {out_channels}, {height}, {width}]")
        print(f"Kernel: {kernel_size}x{kernel_size}, stride={stride}, padding={padding}")
        print(f"{'='*70}")

        # Create input
        x = torch.randn(batch_size, in_channels, height, width, device='cuda')

        # Create PyTorch layers
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                        stride=stride, padding=padding, bias=True).cuda().eval()
        instance_norm = nn.InstanceNorm2d(out_channels, affine=True).cuda().eval()
        relu = nn.ReLU(inplace=False).cuda().eval()

        # Create fused layer
        fused = FusedConvInstanceNormReLU(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        ).cuda().eval()

        # Copy weights for fair comparison
        with torch.no_grad():
            fused.weight.copy_(conv.weight)
            if conv.bias is not None:
                fused.bias.copy_(conv.bias)
            fused.gamma.copy_(instance_norm.weight)
            fused.beta.copy_(instance_norm.bias)

        try:
            # Benchmark PyTorch
            print("\n1. PyTorch Sequential (Conv2d → InstanceNorm2d → ReLU)...")
            pytorch_stats = self.benchmark_pytorch_sequential(x, conv, instance_norm, relu)
            print(f"   {pytorch_stats['mean_ms']:.3f} ± {pytorch_stats['std_ms']:.3f} ms")

            # Get PyTorch output for verification
            with torch.no_grad():
                pytorch_out = conv(x)
                pytorch_out = instance_norm(pytorch_out)
                pytorch_out = relu(pytorch_out)

            # Benchmark Fused
            print("\n2. Fused Conv+InstanceNorm+ReLU Kernel...")
            fused_stats = self.benchmark_fused(x, fused)
            print(f"   {fused_stats['mean_ms']:.3f} ± {fused_stats['std_ms']:.3f} ms")

            # Get fused output for verification
            with torch.no_grad():
                fused_out = fused(x)

            # Correctness check
            print("\n3. Correctness Check...")
            correctness = self.verify_correctness(pytorch_out, fused_out)
            print(f"   Max difference:  {correctness['max_diff']:.2e}")
            print(f"   Mean difference: {correctness['mean_diff']:.2e}")

            if correctness['passed']:
                print(f"   ✓ PASSED (tolerance: {correctness['tolerance']:.0e})")
            else:
                print(f"   ✗ FAILED (tolerance: {correctness['tolerance']:.0e})")

            # Summary
            speedup = pytorch_stats['mean_ms'] / fused_stats['mean_ms']

            print(f"\n{'-'*70}")
            print(f"Speedup: {speedup:.2f}x")

            if speedup < 1.0:
                status = "⚠️  Slower"
            elif speedup < 2.0:
                status = "✓ Modest"
            elif speedup < 5.0:
                status = "✓✓ Good"
            else:
                status = "✓✓✓ Excellent"
            print(f"Status:  {status}")

            result = {
                'name': name,
                'config': {
                    'batch_size': batch_size,
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'height': height,
                    'width': width,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding,
                },
                'pytorch': pytorch_stats,
                'fused': fused_stats,
                'speedup': speedup,
                'correctness': correctness,
                'passed': correctness['passed']
            }

            self.results.append(result)
            return result

        except Exception as e:
            print(f"\n   ❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_summary(self):
        """Print summary of all benchmarks."""
        if not self.results:
            print("\nNo results to display.")
            return

        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)

        print(f"\n{'Config':<25} {'PyTorch (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10}")
        print("-"*70)

        for r in self.results:
            if r is not None:
                pytorch_time = r['pytorch']['mean_ms']
                fused_time = r['fused']['mean_ms']
                speedup = r['speedup']
                passed = "✓" if r['passed'] else "✗"

                print(f"{r['name']:<24} {pytorch_time:<15.3f} {fused_time:<15.3f} "
                      f"{speedup:<10.2f}x {passed}")

        # Calculate average speedup
        speedups = [r['speedup'] for r in self.results if r is not None]
        if speedups:
            avg_speedup = np.mean(speedups)
            print("-"*70)
            print(f"{'Average Speedup':<24} {'':<30} {avg_speedup:<10.2f}x")

        print()


# ============================================================================
# Main Benchmark Functions
# ============================================================================

def run_standard_benchmark():
    """Run standard benchmark suite."""
    print("="*70)
    print("Conv+InstanceNorm+ReLU Fusion - Standard Benchmark")
    print("="*70)

    runner = BenchmarkRunner(warmup=20, iterations=100)

    configs = [
        ("Small feature map", 1, 64, 64, 64, 64, 3, 1, 1),
        ("Medium feature map", 1, 128, 128, 128, 128, 3, 1, 1),
        ("Large feature map", 1, 64, 64, 256, 256, 3, 1, 1),
        ("Residual block", 1, 128, 128, 32, 32, 3, 1, 1),
        ("1x1 bottleneck", 1, 256, 64, 64, 64, 1, 1, 0),
        ("Downsample", 1, 64, 128, 128, 128, 3, 2, 1),
    ]

    for config in configs:
        runner.run_single_benchmark(*config)

    runner.print_summary()

    return runner.results


def run_real_world_benchmark():
    """Benchmark with realistic style transfer network configurations."""
    print("="*70)
    print("Conv+InstanceNorm+ReLU Fusion - Real-World Style Transfer")
    print("="*70)

    runner = BenchmarkRunner(warmup=20, iterations=100)

    # Simulating a typical residual block in Johnson et al. style transfer
    print("\nSimulating typical style transfer network forward pass...")

    # Input layer
    runner.run_single_benchmark(
        "Input layer", 1, 3, 32, 256, 256, 9, 1, 4
    )

    # Encoder blocks
    runner.run_single_benchmark("Encoder block 1", 1, 32, 64, 128, 128, 3, 2, 1)
    runner.run_single_benchmark("Encoder block 2", 1, 64, 128, 64, 64, 3, 2, 1)

    # Residual blocks (typical: 5-10 blocks)
    for i in range(3):
        runner.run_single_benchmark(
            f"Residual block {i+1}", 1, 128, 128, 64, 64, 3, 1, 1
        )

    # Decoder blocks
    runner.run_single_benchmark("Decoder block 1", 1, 128, 64, 128, 128, 3, 1, 1)
    runner.run_single_benchmark("Decoder block 2", 1, 64, 32, 256, 256, 3, 1, 1)

    # Output layer
    runner.run_single_benchmark("Output layer", 1, 32, 3, 256, 256, 9, 1, 4)

    runner.print_summary()

    return runner.results


def run_residual_block_benchmark():
    """Benchmark complete residual blocks."""
    print("="*70)
    print("Residual Block Benchmark (Conv-IN-ReLU x2)")
    print("="*70)

    class PyTorchResidualBlock(nn.Module):
        """PyTorch residual block."""
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.in1 = nn.InstanceNorm2d(channels, affine=True)
            self.relu1 = nn.ReLU(inplace=False)

            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.in2 = nn.InstanceNorm2d(channels, affine=True)
            self.relu2 = nn.ReLU(inplace=False)

        def forward(self, x):
            residual = x
            out = self.relu1(self.in1(self.conv1(x)))
            out = self.relu2(self.in2(self.conv2(out)))
            return out + residual

    # Test with different channel counts
    configs = [64, 128, 256]

    for channels in configs:
        print(f"\n{'='*70}")
        print(f"Residual Block with {channels} channels")
        print(f"{'='*70}")

        x = torch.randn(1, channels, 64, 64, device='cuda')

        # PyTorch version (6 kernel launches: 2 conv, 2 in, 2 relu)
        pytorch_block = PyTorchResidualBlock(channels).cuda().eval()

        # Fused version (2 kernel launches: 2 fused conv-in-relu)
        fused_block = ResidualBlock(channels).cuda().eval()

        # Copy weights
        with torch.no_grad():
            fused_block.conv1.weight.copy_(pytorch_block.conv1.weight)
            fused_block.conv1.bias.copy_(pytorch_block.conv1.bias)
            fused_block.conv1.gamma.copy_(pytorch_block.in1.weight)
            fused_block.conv1.beta.copy_(pytorch_block.in1.bias)

            fused_block.conv2.weight.copy_(pytorch_block.conv2.weight)
            fused_block.conv2.bias.copy_(pytorch_block.conv2.bias)
            fused_block.conv2.gamma.copy_(pytorch_block.in2.weight)
            fused_block.conv2.beta.copy_(pytorch_block.in2.bias)

        # Warmup
        for _ in range(20):
            with torch.no_grad():
                pytorch_out = pytorch_block(x)
                fused_out = fused_block(x)

        torch.cuda.synchronize()

        # Benchmark
        iterations = 100
        times_pytorch = []
        times_fused = []

        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                _ = pytorch_block(x)
            end.record()
            torch.cuda.synchronize()
            times_pytorch.append(start.elapsed_time(end))

            start.record()
            with torch.no_grad():
                _ = fused_block(x)
            end.record()
            torch.cuda.synchronize()
            times_fused.append(start.elapsed_time(end))

        pytorch_mean = np.mean(times_pytorch)
        fused_mean = np.mean(times_fused)
        speedup = pytorch_mean / fused_mean

        print(f"\nPyTorch (6 kernel launches):  {pytorch_mean:.3f} ± {np.std(times_pytorch):.3f} ms")
        print(f"Fused (2 kernel launches):     {fused_mean:.3f} ± {np.std(times_fused):.3f} ms")
        print(f"Speedup: {speedup:.2f}x")

        # Correctness
        with torch.no_grad():
            pytorch_out = pytorch_block(x)
            fused_out = fused_block(x)

        max_diff = torch.max(torch.abs(pytorch_out - fused_out)).item()
        print(f"Max difference: {max_diff:.2e}")

        if max_diff < 1e-4:
            print("✓ Outputs match")
        else:
            print("⚠ Outputs differ")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Benchmark Conv+InstanceNorm+ReLU fusion'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='standard',
        choices=['standard', 'real-world', 'residual', 'all'],
        help='Benchmark mode'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=20,
        help='Number of warmup iterations'
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This benchmark requires a GPU.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    if args.mode == 'standard':
        run_standard_benchmark()
    elif args.mode == 'real-world':
        run_real_world_benchmark()
    elif args.mode == 'residual':
        run_residual_block_benchmark()
    elif args.mode == 'all':
        run_standard_benchmark()
        print("\n" + "="*70 + "\n")
        run_residual_block_benchmark()


if __name__ == "__main__":
    main()
