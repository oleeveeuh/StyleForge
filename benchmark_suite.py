#!/usr/bin/env python3
"""
StyleForge Comprehensive Benchmark Suite

Compares PyTorch baseline vs CUDA kernels across multiple configurations:
- Model types (Fast Style Transfer, ViT)
- Image sizes
- Batch sizes
- Quality metrics

Generates JSON results and comparison plots.
"""

import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import torch

from styleforge_pipeline import (
    StyleForgePipeline,
    PipelineConfig,
    BenchmarkResult,
    QualityMetrics,
    create_pipeline
)


@dataclass
class SuiteConfig:
    """Configuration for benchmark suite."""
    image_sizes: List[int] = None
    iterations: int = 50
    warmup: int = 10
    batch_sizes: List[int] = None
    models: List[str] = None
    save_results: bool = True
    output_dir: str = 'results'

    def __post_init__(self):
        if self.image_sizes is None:
            self.image_sizes = [256, 512, 768, 1024]
        if self.batch_sizes is None:
            self.batch_sizes = [1]
        if self.models is None:
            self.models = ['fast', 'vit_small', 'vit_base']


@dataclass
class ComparisonResult:
    """Result from comparing two backends."""
    model_name: str
    image_size: Tuple[int, int]
    pytorch_time_ms: float
    cuda_time_ms: float
    speedup: float
    pytorch_memory_mb: float
    cuda_memory_mb: float
    memory_reduction: float


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for StyleForge.

    Runs tests comparing:
    - PyTorch baseline (no custom kernels)
    - CUDA accelerated (with custom kernels)

    Across multiple:
    - Image sizes
    - Model types
    - Quality metrics
    """

    def __init__(self, config: Optional[SuiteConfig] = None):
        self.config = config or SuiteConfig()
        self.results: Dict[str, List[BenchmarkResult]] = {}
        self.comparisons: List[ComparisonResult] = []

        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_single_benchmark(
        self,
        model_type: str,
        image_size: int,
        use_cuda: bool,
        iterations: Optional[int] = None
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        iterations = iterations or self.config.iterations

        # Determine backend
        backend = 'cuda' if use_cuda else 'pytorch'

        # Determine model type
        if model_type == 'fast':
            model_config = {'model_type': 'fast', 'backend': backend}
        elif model_type.startswith('vit_'):
            variant = model_type.split('_')[1]
            model_config = {'model_type': 'vit', 'vit_variant': variant, 'backend': backend}
        else:
            model_config = {'model_type': model_type, 'backend': backend}

        # Create pipeline (suppress output)
        config = PipelineConfig(**model_config, verbose=False)
        pipeline = StyleForgePipeline(config)

        # Run benchmark
        result = pipeline.benchmark(
            image_size=image_size,
            iterations=iterations,
            warmup=self.config.warmup,
            collect_memory=True
        )

        # Clean up
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def run_model_comparison(
        self,
        model_type: str,
        image_size: int,
        iterations: Optional[int] = None
    ) -> ComparisonResult:
        """
        Compare PyTorch vs CUDA for a specific model and size.

        Returns:
            ComparisonResult with speedup and memory reduction
        """
        print(f"\n{'='*70}")
        print(f"COMPARISON: {model_type.upper()} @ {image_size}x{image_size}")
        print(f"{'='*70}")

        # Run PyTorch baseline
        print("\n[1/2] Running PyTorch baseline...")
        pytorch_result = self.run_single_benchmark(
            model_type=model_type,
            image_size=image_size,
            use_cuda=False,
            iterations=iterations
        )

        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("\n⚠️  CUDA not available, skipping CUDA benchmark")
            return ComparisonResult(
                model_name=model_type,
                image_size=(image_size, image_size),
                pytorch_time_ms=pytorch_result.avg_time_ms,
                cuda_time_ms=pytorch_result.avg_time_ms,
                speedup=1.0,
                pytorch_memory_mb=pytorch_result.memory_mb,
                cuda_memory_mb=pytorch_result.memory_mb,
                memory_reduction=0.0
            )

        # Run CUDA benchmark
        print("\n[2/2] Running CUDA with custom kernels...")
        cuda_result = self.run_single_benchmark(
            model_type=model_type,
            image_size=image_size,
            use_cuda=True,
            iterations=iterations
        )

        # Calculate comparison metrics
        speedup = pytorch_result.avg_time_ms / cuda_result.avg_time_ms
        memory_reduction = (1 - cuda_result.memory_mb / max(pytorch_result.memory_mb, 1)) * 100

        comparison = ComparisonResult(
            model_name=model_type,
            image_size=(image_size, image_size),
            pytorch_time_ms=pytorch_result.avg_time_ms,
            cuda_time_ms=cuda_result.avg_time_ms,
            speedup=speedup,
            pytorch_memory_mb=pytorch_result.memory_mb,
            cuda_memory_mb=cuda_result.memory_mb,
            memory_reduction=memory_reduction
        )

        # Print comparison
        print(f"\n{'='*70}")
        print("COMPARISON RESULTS")
        print(f"{'='*70}")
        print(f"PyTorch: {pytorch_result.avg_time_ms:.2f} ms")
        print(f"CUDA:    {cuda_result.avg_time_ms:.2f} ms")
        print(f"\n🚀 Speedup: {speedup:.2f}x")
        print(f"📊 Memory: {cuda_result.memory_mb:.2f} MB vs {pytorch_result.memory_mb:.2f} MB")
        print(f"📉 Memory Reduction: {memory_reduction:.1f}%")

        # Kernel-specific stats
        if cuda_result.cuda_kernel_calls > 0:
            print(f"\n🔧 CUDA Kernel Calls: {cuda_result.cuda_kernel_calls}")
            print(f"🔧 CUDA Usage: {cuda_result.cuda_percentage:.1f}%")

        self.comparisons.append(comparison)

        return comparison

    def run_full_suite(self) -> Dict[str, List[ComparisonResult]]:
        """
        Run full benchmark suite across all configurations.

        Returns:
            Dictionary mapping model names to comparison results
        """
        print(f"\n{'#'*70}")
        print(f"# STYLEFORGE BENCHMARK SUITE")
        print(f"{'#'*70}")
        print(f"\nConfiguration:")
        print(f"  Image sizes: {self.config.image_sizes}")
        print(f"  Models: {self.config.models}")
        print(f"  Iterations: {self.config.iterations}")
        print(f"  Warmup: {self.config.warmup}")
        print(f"\nCUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"{'#'*70}")

        all_results: Dict[str, List[ComparisonResult]] = {}

        for model_type in self.config.models:
            model_results = []

            for size in self.config.image_sizes:
                try:
                    result = self.run_model_comparison(
                        model_type=model_type,
                        image_size=size,
                        iterations=self.config.iterations
                    )
                    model_results.append(result)
                except Exception as e:
                    print(f"\n❌ Error benchmarking {model_type} at {size}x{size}: {e}")
                    continue

            all_results[model_type] = model_results

        # Save results
        if self.config.save_results:
            self.save_results(all_results)

        # Print summary
        self.print_summary(all_results)

        return all_results

    def run_quality_comparison(
        self,
        model_type: str = 'vit_small',
        image_size: int = 512,
        num_samples: int = 5
    ) -> List[QualityMetrics]:
        """
        Compare output quality between PyTorch and CUDA backends.

        Generates multiple random inputs and compares outputs to ensure
        numerical correctness.
        """
        print(f"\n{'='*70}")
        print("QUALITY COMPARISON")
        print(f"{'='*70}")
        print(f"Model: {model_type}")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Samples: {num_samples}")

        quality_results = []

        # Create both pipelines
        pytorch_config = PipelineConfig(
            model_type='vit' if model_type.startswith('vit') else model_type,
            backend='pytorch',
            verbose=False
        )
        cuda_config = PipelineConfig(
            model_type='vit' if model_type.startswith('vit') else model_type,
            backend='cuda',
            verbose=False
        )

        pytorch_pipeline = StyleForgePipeline(pytorch_config)
        cuda_pipeline = StyleForgePipeline(cuda_config)

        for i in range(num_samples):
            print(f"\n[{i+1}/{num_samples}]")

            # Generate random inputs
            content = torch.randn(1, 3, image_size, image_size)
            style = torch.randn(1, 3, image_size, image_size)

            # Run both pipelines
            pytorch_pipeline.model.eval()
            cuda_pipeline.model.eval()

            with torch.no_grad():
                pytorch_output = pytorch_pipeline.model(content, style)
                cuda_output = cuda_pipeline.model(content, style)

            # Compare quality
            quality = pytorch_pipeline.compare_quality(pytorch_output, cuda_output)
            quality_results.append(quality)

            print(f"  SSIM: {quality.ssim:.4f}")
            print(f"  PSNR: {quality.psnr:.2f} dB")
            print(f"  MSE:  {quality.mse:.6e}")
            print(f"  MAE:  {quality.mae:.6e}")

        # Calculate averages
        avg_ssim = np.mean([q.ssim for q in quality_results])
        avg_psnr = np.mean([q.psnr for q in quality_results])
        avg_mse = np.mean([q.mse for q in quality_results])
        avg_mae = np.mean([q.mae for q in quality_results])

        print(f"\n{'='*70}")
        print("AVERAGE QUALITY METRICS")
        print(f"{'='*70}")
        print(f"SSIM: {avg_ssim:.4f} (1.0 = perfect)")
        print(f"PSNR: {avg_psnr:.2f} dB (higher is better)")
        print(f"MSE:  {avg_mse:.6e} (lower is better)")
        print(f"MAE:  {avg_mae:.6e} (lower is better)")

        # Quality check
        if avg_ssim > 0.99 and avg_psnr > 40:
            print("\n✅ PASS: CUDA implementation matches PyTorch baseline")
        elif avg_ssim > 0.95:
            print("\n⚠️  WARNING: Minor numerical differences detected")
        else:
            print("\n❌ FAIL: Significant numerical differences - check implementation")

        return quality_results

    def save_results(self, all_results: Dict[str, List[ComparisonResult]]):
        """Save benchmark results to JSON file."""
        results_path = self.output_dir / 'benchmark_results.json'

        # Convert to serializable format
        serializable = {}
        for model, results in all_results.items():
            serializable[model] = [asdict(r) for r in results]

        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\n✅ Results saved to {results_path}")

    def print_summary(self, all_results: Dict[str, List[ComparisonResult]]):
        """Print summary of all benchmark results."""
        print(f"\n{'#'*70}")
        print(f"# BENCHMARK SUITE SUMMARY")
        print(f"{'#'*70}")

        for model, results in all_results.items():
            if not results:
                continue

            print(f"\n{model.upper()}")
            print("-" * 70)
            print(f"{'Size':<10} {'PyTorch':<12} {'CUDA':<12} {'Speedup':<10}")
            print("-" * 70)

            for r in results:
                size_str = f"{r.image_size[0]}x{r.image_size[1]}"
                print(f"{size_str:<10} {r.pytorch_time_ms:<10.2f}ms "
                      f"{r.cuda_time_ms:<10.2f}ms {r.speedup:<10.2f}x")

        # Overall statistics
        all_speedups = [r.speedup for results in all_results.values() for r in results if r.speedup > 1]
        if all_speedups:
            avg_speedup = np.mean(all_speedups)
            max_speedup = np.max(all_speedups)

            print(f"\n{'='*70}")
            print(f"Overall Statistics:")
            print(f"  Average Speedup: {avg_speedup:.2f}x")
            print(f"  Maximum Speedup: {max_speedup:.2f}x")
            print(f"  Configurations Tested: {len(all_speedups)}")
            print(f"{'='*70}")

    def generate_plots(self, all_results: Dict[str, List[ComparisonResult]]):
        """Generate comparison plots (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n⚠️  matplotlib not available, skipping plots")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('StyleForge Benchmark Results', fontsize=16)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Plot 1: Speedup by image size
        ax = axes[0, 0]
        x_indices = np.arange(len(self.config.image_sizes))
        width = 0.8 / len(all_results)

        for i, (model, results) in enumerate(all_results.items()):
            if not results:
                continue
            sizes = [r.image_size[0] for r in results]
            speedups = [r.speedup for r in results]
            ax.plot(sizes, speedups, marker='o', label=model, color=colors[i % len(colors)])

        ax.set_xlabel('Image Size')
        ax.set_ylabel('Speedup (x)')
        ax.set_title('Speedup vs Image Size')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Time comparison (bar)
        ax = axes[0, 1]
        models = list(all_results.keys())
        pytorch_times = [np.mean([r.pytorch_time_ms for r in all_results[m]]) for m in models]
        cuda_times = [np.mean([r.cuda_time_ms for r in all_results[m]]) for m in models]

        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, pytorch_times, width, label='PyTorch', alpha=0.8)
        ax.bar(x + width/2, cuda_times, width, label='CUDA', alpha=0.8)
        ax.set_ylabel('Time (ms)')
        ax.set_title('Average Time by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        # Plot 3: FPS comparison
        ax = axes[1, 0]
        pytorch_fps = [1000 / t for t in pytorch_times]
        cuda_fps = [1000 / t for t in cuda_times]

        ax.bar(x - width/2, pytorch_fps, width, label='PyTorch', alpha=0.8)
        ax.bar(x + width/2, cuda_fps, width, label='CUDA', alpha=0.8)
        ax.set_ylabel('FPS')
        ax.set_title('Throughput by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        # Plot 4: Memory usage
        ax = axes[1, 1]
        pytorch_mem = [np.mean([r.pytorch_memory_mb for r in all_results[m]]) for m in models]
        cuda_mem = [np.mean([r.cuda_memory_mb for r in all_results[m]]) for m in models]

        ax.bar(x - width/2, pytorch_mem, width, label='PyTorch', alpha=0.8)
        ax.bar(x + width/2, cuda_mem, width, label='CUDA', alpha=0.8)
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / 'benchmark_plots.png'
        plt.savefig(plot_path, dpi=150)
        print(f"\n✅ Plots saved to {plot_path}")

        return plot_path


def run_quick_benchmark() -> Dict:
    """Run a quick benchmark with default settings."""
    config = SuiteConfig(
        image_sizes=[256, 512],
        iterations=20,
        models=['fast', 'vit_small']
    )
    suite = BenchmarkSuite(config)
    return suite.run_full_suite()


def run_full_benchmark() -> Dict:
    """Run comprehensive benchmark suite."""
    config = SuiteConfig(
        image_sizes=[256, 512, 768, 1024],
        iterations=50,
        models=['fast', 'vit_small', 'vit_base']
    )
    suite = BenchmarkSuite(config)
    results = suite.run_full_suite()
    suite.generate_plots(results)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StyleForge Benchmark Suite")
    parser.add_argument('--mode', choices=['quick', 'full', 'quality'],
                        default='quick', help='Benchmark mode')
    parser.add_argument('--sizes', nargs='+', type=int,
                        default=[256, 512], help='Image sizes to test')
    parser.add_argument('--models', nargs='+',
                        default=['fast', 'vit_small'], help='Models to test')
    parser.add_argument('--iters', type=int, default=50, help='Iterations per benchmark')
    parser.add_argument('--quality-samples', type=int, default=5,
                        help='Number of samples for quality comparison')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')

    args = parser.parse_args()

    config = SuiteConfig(
        image_sizes=args.sizes,
        iterations=args.iters,
        models=args.models,
        output_dir=args.output_dir
    )

    suite = BenchmarkSuite(config)

    if args.mode == 'quality':
        suite.run_quality_comparison(
            model_type=args.models[0] if args.models else 'vit_small',
            image_size=args.sizes[0] if args.sizes else 512,
            num_samples=args.quality_samples
        )
    else:
        results = suite.run_full_suite()

        if not args.no_plots:
            suite.generate_plots(results)
