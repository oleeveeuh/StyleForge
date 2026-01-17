#!/usr/bin/env python3
"""
StyleForge - Full Benchmark Suite

Run comprehensive benchmarks using the benchmarking framework.
This script demonstrates the complete workflow:
1. Configure benchmarks
2. Run comparisons
3. Validate correctness
4. Generate reports
5. Create visualizations
"""

import torch
import torch.nn as nn
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarking import (
    BenchmarkFramework,
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkVisualizer,
    HAS_MATPLOTLIB,
)

# Import kernels to benchmark
try:
    from kernels.instance_norm_wrapper import FusedInstanceNorm2d
    HAS_INSTANCE_NORM = True
except ImportError:
    HAS_INSTANCE_NORM = False
    print("⚠️  InstanceNorm kernel not available")

try:
    from kernels.ffn_wrapper import FusedFFN
    HAS_FFN = True
except ImportError:
    HAS_FFN = False
    print("⚠️  FFN kernel not available")


# ============================================================================
# Benchmark Suite Functions
# ============================================================================

def run_instance_norm_suite(
    framework: BenchmarkFramework,
    output_dir: str
):
    """Run InstanceNorm benchmark suite."""
    if not HAS_INSTANCE_NORM:
        print("\n⚠️  Skipping InstanceNorm benchmarks (kernel not available)")
        return []

    print("\n" + "=" * 70)
    print("Instance Normalization Benchmark Suite")
    print("=" * 70)

    # Define configurations
    configs = [
        BenchmarkConfig("64×64", 1, 64, 64, 64, iterations=100),
        BenchmarkConfig("128×128", 1, 128, 128, 128, iterations=100),
        BenchmarkConfig("256×256", 1, 64, 256, 256, iterations=50),
        BenchmarkConfig("32×32 residual", 1, 128, 32, 32, iterations=100),
        BenchmarkConfig("Batch 4×64×64", 4, 64, 64, 64, iterations=100),
    ]

    all_results = []

    for config in configs:
        # Create input tensor
        x = framework.create_input_tensor(config)

        # Create layers
        pytorch_layer = nn.InstanceNorm2d(config.channels, affine=True).cuda().eval()
        fused_layer = FusedInstanceNorm2d(config.channels, affine=True).cuda().eval()

        # Copy weights
        with torch.no_grad():
            fused_layer.gamma.copy_(pytorch_layer.weight)
            fused_layer.beta.copy_(pytorch_layer.bias)

        # Define functions
        def baseline_func(inp):
            return pytorch_layer(inp)

        def optimized_func(inp):
            return fused_layer(inp)

        # Run comparison
        result = framework.compare(
            baseline_func=baseline_func,
            optimized_func=optimized_func,
            config=config,
            input_tensor=x,
            validate=True,
            verbose=True
        )

        if result is not None:
            all_results.append(result.to_dict())

    return all_results


def run_ffn_suite(
    framework: BenchmarkFramework,
    output_dir: str
):
    """Run FFN benchmark suite."""
    if not HAS_FFN:
        print("\n⚠️  Skipping FFN benchmarks (kernel not available)")
        return []

    print("\n" + "=" * 70)
    print("Feed-Forward Network Benchmark Suite")
    print("=" * 70)

    # Define configurations
    configs = [
        BenchmarkConfig("Small (128)", 2, 256, 128, 128, iterations=100),
        BenchmarkConfig("Medium (512)", 2, 512, 64, 64, iterations=100),
        BenchmarkConfig("Large (1024)", 2, 1024, 32, 32, iterations=50),
    ]

    all_results = []

    for config in configs:
        # Create layers
        embed_dim = config.channels
        ffn_dim = embed_dim * 4

        pytorch_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        ).cuda().eval()

        fused_ffn = FusedFFN(embed_dim, ffn_dim).cuda().eval()

        # Copy weights
        with torch.no_grad():
            # Map PyTorch weights to fused format
            fused_ffn.fc1_weight.data.copy_(pytorch_ffn[0].weight.data.T)
            fused_ffn.fc1_bias.data.copy_(pytorch_ffn[0].bias.data)
            fused_ffn.fc2_weight.data.copy_(pytorch_ffn[2].weight.data.T)
            fused_ffn.fc2_bias.data.copy_(pytorch_ffn[2].bias.data)

        # Create input (3D for FFN)
        x = torch.randn(
            config.batch_size,
            config.height,  # seq_len
            config.width,   # embed_dim
            device='cuda'
        )

        # Define functions
        def baseline_func(inp):
            out = pytorch_ffn(inp)
            return out  # Return 3D tensor

        def optimized_func(inp):
            return fused_ffn(inp)

        # Manual benchmark (different config for 3D)
        # Run validation first
        with torch.no_grad():
            pytorch_out = baseline_func(x)
            fused_out = optimized_func(x)

        max_error = torch.max(torch.abs(pytorch_out - fused_out)).item()
        is_correct = torch.allclose(pytorch_out, fused_out, rtol=1e-4, atol=1e-6)

        print(f"\n{'='*70}")
        print(f"Benchmark: {config.name}")
        print(f"Shape: {x.shape}")
        print(f"{'='*70}")

        if is_correct:
            print(f"  Validation: ✓ PASSED (max error: {max_error:.2e})")
        else:
            print(f"  Validation: ✗ FAILED (max error: {max_error:.2e})")
            continue

        # Benchmark baseline
        baseline_result = framework.benchmark_function(
            baseline_func, config, x,
            description=f"{config.name} (Baseline)"
        )
        print(f"  Baseline:  {baseline_result.mean_ms:.4f} ± {baseline_result.std_ms:.4f} ms")

        # Benchmark optimized
        optimized_result = framework.benchmark_function(
            optimized_func, config, x,
            description=f"{config.name} (Optimized)"
        )
        print(f"  Optimized: {optimized_result.mean_ms:.4f} ± {optimized_result.std_ms:.4f} ms")

        speedup = baseline_result.mean_ms / optimized_result.mean_ms
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"  Status:    {framework._get_speedup_status(speedup)}")

        # Create comparison result manually
        from benchmarking.benchmark_framework import ComparisonResult

        comparison = ComparisonResult(
            config=config.name,
            baseline=baseline_result,
            optimized=optimized_result,
            speedup=speedup,
            max_error=max_error,
            passed_validation=is_correct,
            memory_transfer_reduction=framework._estimate_memory_reduction(speedup),
            timestamp=baseline_result.timestamp
        )

        all_results.append(comparison.to_dict())

    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    """Run full benchmark suite."""
    parser = argparse.ArgumentParser(
        description='StyleForge Full Benchmark Suite'
    )
    parser.add_argument(
        '--kernels',
        type=str,
        default='all',
        help='Kernels to benchmark: instance_norm, ffn, all'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations'
    )
    parser.add_argument(
        '--no-charts',
        action='store_true',
        help='Skip chart generation'
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Benchmarks require a GPU.")
        return 1

    print("=" * 70)
    print("StyleForge Benchmark Suite")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create framework
    framework = BenchmarkFramework(use_cuda_events=True)

    # Run benchmarks
    all_results = []

    if args.kernels in ['all', 'instance_norm']:
        results = run_instance_norm_suite(framework, str(output_dir))
        all_results.extend(results)

    if args.kernels in ['all', 'ffn']:
        results = run_ffn_suite(framework, str(output_dir))
        all_results.extend(results)

    if not all_results:
        print("\n⚠️  No benchmarks were run successfully")
        return 1

    # Save raw results
    framework.save_results(output_dir / 'raw_results.json')

    # Print summary
    framework.print_summary()

    # Generate reports
    print("\n" + "=" * 70)
    print("Generating Reports")
    print("=" * 70)

    report = BenchmarkReport(
        title="StyleForge CUDA Kernel Benchmark Report",
        subtitle="Performance Analysis of Fused Kernels"
    )

    report.generate_all_formats(all_results, str(output_dir))

    # Generate charts
    if not args.no_charts:
        print("\n" + "=" * 70)
        print("Generating Charts")
        print("=" * 70)

        if HAS_MATPLOTLIB:
            visualizer = BenchmarkVisualizer()
            charts_dir = output_dir / 'charts'
            visualizer.generate_all_charts(all_results, str(charts_dir))
        else:
            print("⚠️  Charts skipped (matplotlib not installed)")
            print("   Install with: pip install matplotlib")

    # Final summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - raw_results.json")
    print(f"  - report.md")
    print(f"  - report.html")
    print(f"  - report.csv")
    if not args.no_charts and HAS_MATPLOTLIB:
        print(f"  - charts/*.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
