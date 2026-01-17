#!/usr/bin/env python3
"""
StyleForge - Comprehensive Instance Normalization Benchmark

Benchmark script for comparing fused InstanceNorm CUDA kernel
against PyTorch's nn.InstanceNorm2d.

This is a critical operation in style transfer networks, appearing
15-20 times per forward pass.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from kernels.instance_norm_wrapper import FusedInstanceNorm2d


# ============================================================================
# Benchmark Runner
# ============================================================================

class InstanceNormBenchmarkRunner:
    """Run comprehensive benchmarks for InstanceNorm."""

    def __init__(self, warmup: int = 20, iterations: int = 100):
        self.warmup = warmup
        self.iterations = iterations
        self.results = []

    def benchmark_pytorch(
        self,
        x: torch.Tensor,
        layer: nn.InstanceNorm2d
    ) -> Dict[str, float]:
        """Benchmark PyTorch InstanceNorm2d."""
        # Warmup
        for _ in range(self.warmup):
            with torch.no_grad():
                _ = layer(x)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(self.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                _ = layer(x)
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
        layer: FusedInstanceNorm2d
    ) -> Dict[str, float]:
        """Benchmark fused InstanceNorm."""
        # Warmup
        for _ in range(self.warmup):
            with torch.no_grad():
                _ = layer(x)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(self.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                _ = layer(x)
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
        """Verify outputs match within tolerance."""
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
        channels: int,
        height: int,
        width: int,
        use_vectorized: bool = True
    ) -> Dict[str, Any]:
        """Run a single benchmark configuration."""
        print(f"\n{'='*70}")
        print(f"{name}: [{batch_size}, {channels}, {height}, {width}]")
        print(f"{'='*70}")

        # Create input
        x = torch.randn(batch_size, channels, height, width, device='cuda')

        # Create PyTorch layer
        pytorch_layer = nn.InstanceNorm2d(channels, affine=True).cuda().eval()

        # Create fused layer
        fused_layer = FusedInstanceNorm2d(
            channels, affine=True, use_vectorized=use_vectorized
        ).cuda().eval()

        # Copy weights for fair comparison
        with torch.no_grad():
            fused_layer.gamma.copy_(pytorch_layer.weight)
            fused_layer.beta.copy_(pytorch_layer.bias)

        try:
            # Benchmark PyTorch
            print("\n1. PyTorch InstanceNorm2d...")
            pytorch_stats = self.benchmark_pytorch(x, pytorch_layer)
            print(f"   {pytorch_stats['mean_ms']:.4f} ± {pytorch_stats['std_ms']:.4f} ms")

            # Get PyTorch output for verification
            with torch.no_grad():
                pytorch_out = pytorch_layer(x)

            # Benchmark Fused
            vec_str = " (vectorized)" if use_vectorized else " (scalar)"
            print(f"\n2. Fused InstanceNorm{vec_str}...")
            fused_stats = self.benchmark_fused(x, fused_layer)
            print(f"   {fused_stats['mean_ms']:.4f} ± {fused_stats['std_ms']:.4f} ms")

            # Get fused output for verification
            with torch.no_grad():
                fused_out = fused_layer(x)

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
            elif speedup < 3.0:
                status = "✓✓ Good"
            else:
                status = "✓✓✓ Excellent"
            print(f"Status:  {status}")

            result = {
                'name': name,
                'config': {
                    'batch_size': batch_size,
                    'channels': channels,
                    'height': height,
                    'width': width,
                    'use_vectorized': use_vectorized,
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

                print(f"{r['name']:<24} {pytorch_time:<15.4f} {fused_time:<15.4f} "
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
    print("Instance Normalization - Standard Benchmark")
    print("="*70)

    runner = InstanceNormBenchmarkRunner(warmup=20, iterations=100)

    configs = [
        ("Small (64x64)", 1, 64, 64, 64),
        ("Medium (128x128)", 1, 128, 128, 128),
        ("Large (256x256)", 1, 64, 256, 256),
        ("Residual (32x32)", 1, 128, 32, 32),
        ("Tiny (16x16)", 1, 256, 16, 16),
        ("Batch (4x64x64)", 4, 64, 64, 64),
    ]

    for config in configs:
        runner.run_single_benchmark(*config)

    runner.print_summary()

    return runner.results


def run_vectorized_comparison():
    """Compare vectorized vs scalar implementations."""
    print("="*70)
    print("Instance Normalization - Vectorized vs Scalar")
    print("="*70)

    runner = InstanceNormBenchmarkRunner(warmup=20, iterations=100)

    # Test with feature map that has spatial_size % 4 == 0
    name = "Vectorized test (128x128)"
    batch, channels, h, w = 1, 64, 128, 128

    print(f"\n{'='*70}")
    print(f"{name}: [{batch}, {channels}, {h}, {w}]")
    print(f"{'='*70}")

    x = torch.randn(batch, channels, h, w, device='cuda')
    pytorch_layer = nn.InstanceNorm2d(channels, affine=True).cuda().eval()

    # PyTorch baseline
    print("\n1. PyTorch InstanceNorm2d...")
    pytorch_stats = runner.benchmark_pytorch(x, pytorch_layer)
    print(f"   {pytorch_stats['mean_ms']:.4f} ± {pytorch_stats['std_ms']:.4f} ms")

    with torch.no_grad():
        pytorch_out = pytorch_layer(x)

    # Fused scalar
    print("\n2. Fused InstanceNorm (scalar)...")
    fused_scalar = FusedInstanceNorm2d(channels, affine=True, use_vectorized=False).cuda().eval()
    with torch.no_grad():
        fused_scalar.gamma.copy_(pytorch_layer.weight)
        fused_scalar.beta.copy_(pytorch_layer.bias)

    scalar_stats = runner.benchmark_fused(x, fused_scalar)
    print(f"   {scalar_stats['mean_ms']:.4f} ± {scalar_stats['std_ms']:.4f} ms")

    with torch.no_grad():
        scalar_out = fused_scalar(x)

    # Fused vectorized
    print("\n3. Fused InstanceNorm (vectorized)...")
    fused_vec = FusedInstanceNorm2d(channels, affine=True, use_vectorized=True).cuda().eval()
    with torch.no_grad():
        fused_vec.gamma.copy_(pytorch_layer.weight)
        fused_vec.beta.copy_(pytorch_layer.bias)

    vec_stats = runner.benchmark_fused(x, fused_vec)
    print(f"   {vec_stats['mean_ms']:.4f} ± {vec_stats['std_ms']:.4f} ms")

    with torch.no_grad():
        vec_out = fused_vec(x)

    # Correctness
    print("\n4. Correctness Check...")
    scalar_diff = torch.max(torch.abs(pytorch_out - scalar_out)).item()
    vec_diff = torch.max(torch.abs(pytorch_out - vec_out)).item()

    print(f"   Scalar max diff:  {scalar_diff:.2e}")
    print(f"   Vectorized max diff: {vec_diff:.2e}")

    if scalar_diff < 1e-4 and vec_diff < 1e-4:
        print("   ✓ Both implementations match PyTorch")
    else:
        print("   ✗ Outputs differ from PyTorch")

    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)

    scalar_speedup = pytorch_stats['mean_ms'] / scalar_stats['mean_ms']
    vec_speedup = pytorch_stats['mean_ms'] / vec_stats['mean_ms']

    print(f"\nPyTorch baseline:       {pytorch_stats['mean_ms']:.4f} ms")
    print(f"Fused (scalar):         {scalar_stats['mean_ms']:.4f} ms ({scalar_speedup:.2f}x)")
    print(f"Fused (vectorized):     {vec_stats['mean_ms']:.4f} ms ({vec_speedup:.2f}x)")

    vec_vs_scalar = scalar_stats['mean_ms'] / vec_stats['mean_ms']
    print(f"\nVectorized is {vec_vs_scalar:.2f}x faster than scalar")

    return {
        'pytorch': pytorch_stats,
        'scalar': scalar_stats,
        'vectorized': vec_stats,
        'scalar_speedup': scalar_speedup,
        'vec_speedup': vec_speedup,
        'vec_vs_scalar': vec_vs_scalar,
    }


def run_style_transfer_simulation():
    """Simulate a typical style transfer network forward pass."""
    print("="*70)
    print("Instance Normalization - Style Transfer Network Simulation")
    print("="*70)
    print("\nSimulating Johnson et al. style transfer architecture...")
    print("InstanceNorm appears at each layer in the network.")

    runner = InstanceNormBenchmarkRunner(warmup=20, iterations=100)

    # Typical style transfer network layers with InstanceNorm
    layers = [
        ("Input encoder", 1, 32, 256, 256),
        ("Encoder block 1", 1, 64, 128, 128),
        ("Encoder block 2", 1, 128, 64, 64),
        ("Residual block 1", 1, 128, 64, 64),
        ("Residual block 2", 1, 128, 64, 64),
        ("Residual block 3", 1, 128, 64, 64),
        ("Decoder block 1", 1, 128, 64, 64),
        ("Decoder block 2", 1, 64, 128, 128),
        ("Decoder block 3", 1, 32, 256, 256),
    ]

    total_pytorch_time = 0
    total_fused_time = 0

    for name, batch, channels, h, w in layers:
        result = runner.run_single_benchmark(name, batch, channels, h, w)
        if result:
            total_pytorch_time += result['pytorch']['mean_ms']
            total_fused_time += result['fused']['mean_ms']

    # Network summary
    print("\n" + "="*70)
    print("NETWORK-WIDE SUMMARY")
    print("="*70)

    print(f"\nTotal InstanceNorm time per forward pass:")
    print(f"  PyTorch:  {total_pytorch_time:.4f} ms")
    print(f"  Fused:    {total_fused_time:.4f} ms")
    print(f"  Saved:    {total_pytorch_time - total_fused_time:.4f} ms")

    overall_speedup = total_pytorch_time / total_fused_time
    print(f"\nOverall speedup: {overall_speedup:.2f}x")

    # At 30 FPS (33.33ms per frame)
    frame_budget = 33.33
    pytorch_pct = (total_pytorch_time / frame_budget) * 100
    fused_pct = (total_fused_time / frame_budget) * 100

    print(f"\nAt 30 FPS (33.33ms/frame budget):")
    print(f"  PyTorch InstanceNorm:  {pytorch_pct:.1f}% of frame budget")
    print(f"  Fused InstanceNorm:    {fused_pct:.1f}% of frame budget")

    savings_pct = ((total_pytorch_time - total_fused_time) / frame_budget) * 100
    print(f"  Budget saved:          {savings_pct:.1f}%")

    return runner.results


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark InstanceNorm fusion'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='standard',
        choices=['standard', 'vectorized', 'style-transfer', 'all'],
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
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    if args.mode == 'standard':
        run_standard_benchmark()
    elif args.mode == 'vectorized':
        run_vectorized_comparison()
    elif args.mode == 'style-transfer':
        run_style_transfer_simulation()
    elif args.mode == 'all':
        run_standard_benchmark()
        print("\n" + "="*70 + "\n")
        run_vectorized_comparison()
        print("\n" + "="*70 + "\n")
        run_style_transfer_simulation()

    return 0


if __name__ == "__main__":
    sys.exit(main())
