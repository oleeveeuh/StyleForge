"""
StyleForge - Benchmark Visualization

Create visualizations for benchmark results including
latency distributions, time series, and comparison plots.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class BenchmarkVisualizer:
    """
    Create visualizations for benchmark results

    Supports:
    - Latency distribution histograms
    - Latency over time plots
    - Summary statistics tables
    - Comparison plots between multiple runs
    """

    def __init__(self, save_dir: str | Path = "benchmarks"):
        """
        Args:
            save_dir: Directory to save visualization images
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_baseline_results(
        self,
        times_ms: np.ndarray,
        result: 'BenchmarkResult',
        save_path: Optional[str | Path] = None
    ):
        """
        Create comprehensive baseline benchmark visualization

        Args:
            times_ms: Raw latency times array
            result: BenchmarkResult with statistics
            save_path: Optional custom save path
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Latency distribution (histogram)
        ax1 = axes[0]
        ax1.hist(times_ms, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(result.latency_ms, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {result.latency_ms:.2f}ms')
        ax1.axvline(result.p95_ms, color='orange', linestyle='--', linewidth=2,
                    label=f'P95: {result.p95_ms:.2f}ms')
        ax1.set_xlabel('Latency (ms)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Latency Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Latency over time
        ax2 = axes[1]
        ax2.plot(times_ms, linewidth=1, alpha=0.7, color='steelblue')
        ax2.axhline(result.latency_ms, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {result.latency_ms:.2f}ms')
        ax2.fill_between(range(len(times_ms)),
                         result.latency_ms - result.std_ms,
                         result.latency_ms + result.std_ms,
                         alpha=0.2, color='red', label='±1 STD')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Latency (ms)', fontsize=12)
        ax2.set_title('Latency Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Summary statistics
        ax3 = axes[2]
        ax3.axis('off')

        summary_text = self._format_summary_box(result)
        ax3.text(0.05, 0.5, summary_text,
                 fontsize=11,
                 family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        if save_path is None:
            save_path = self.save_dir / "baseline_visualization.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_comparison(
        self,
        results: List['BenchmarkResult'],
        baseline_name: str = "PyTorch Baseline",
        save_path: Optional[str | Path] = None
    ):
        """
        Create comparison plot for multiple benchmark results

        Args:
            results: List of BenchmarkResult objects
            baseline_name: Name of baseline for comparison
            save_path: Optional custom save path
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        names = [r.name for r in results]
        latencies = [r.latency_ms for r in results]
        fps_values = [r.fps for r in results]

        # Find baseline index
        baseline_idx = next((i for i, r in enumerate(results) if r.name == baseline_name), 0)
        baseline_latency = latencies[baseline_idx]

        # Calculate speedups
        speedups = [baseline_latency / lat for lat in latencies]

        # 1. Latency comparison
        ax1 = axes[0]
        colors = ['steelblue' if i == baseline_idx else 'coral' for i in range(len(results))]
        bars = ax1.barh(names, latencies, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Latency (ms)', fontsize=12)
        ax1.set_title('Latency Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, val in zip(bars, latencies):
            ax1.text(val, bar.get_y() + bar.get_height()/2,
                    f' {val:.1f}ms', va='center', fontsize=10)

        # 2. Speedup comparison
        ax2 = axes[1]
        colors = ['steelblue' if i == baseline_idx else 'coral' for i in range(len(results))]
        bars = ax2.barh(names, speedups, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Speedup (x)', fontsize=12)
        ax2.set_title('Speedup vs Baseline', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, val in zip(bars, speedups):
            ax2.text(val, bar.get_y() + bar.get_height()/2,
                    f' {val:.2f}x', va='center', fontsize=10)

        # Add baseline line
        ax2.axvline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=2)

        plt.tight_layout()

        if save_path is None:
            save_path = self.save_dir / "comparison_visualization.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_optimization_progress(
        self,
        results: List[Dict[str, Any]],
        save_path: Optional[str | Path] = None
    ):
        """
        Plot optimization progress over multiple iterations

        Args:
            results: List of result dictionaries with 'name' and 'latency_ms'
            save_path: Optional custom save path
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        iterations = list(range(1, len(results) + 1))
        latencies = [r['latency_ms'] for r in results]
        speedups = [results[0]['latency_ms'] / lat for lat in latencies]

        # 1. Latency over iterations
        ax1 = axes[0]
        ax1.plot(iterations, latencies, marker='o', linewidth=2, markersize=8,
                 color='steelblue', label='Latency')
        ax1.fill_between(iterations, 0, latencies, alpha=0.2, color='steelblue')
        ax1.set_xlabel('Optimization Step', fontsize=12)
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_title('Latency Reduction', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add step labels
        for i, (x, y, name) in enumerate(zip(iterations, latencies, [r['name'] for r in results])):
            ax1.annotate(name.split()[-1], (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9)

        # 2. Speedup over iterations
        ax2 = axes[1]
        ax2.plot(iterations, speedups, marker='o', linewidth=2, markersize=8,
                 color='coral', label='Speedup')
        ax2.fill_between(iterations, 1, speedups, alpha=0.2, color='coral')
        ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Optimization Step', fontsize=12)
        ax2.set_ylabel('Speedup (x)', fontsize=12)
        ax2.set_title('Cumulative Speedup', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add step labels
        for i, (x, y, name) in enumerate(zip(iterations, speedups, [r['name'] for r in results])):
            ax2.annotate(name.split()[-1], (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9)

        plt.tight_layout()

        if save_path is None:
            save_path = self.save_dir / "optimization_progress.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def _format_summary_box(self, result: 'BenchmarkResult') -> str:
        """Format summary statistics as text box"""
        return f"""
╔══════════════════════════════════════╗
║     PERFORMANCE SUMMARY              ║
╠══════════════════════════════════════╣
║                                      ║
║  Mean Latency:  {result.latency_ms:>10.2f} ms     ║
║  Std Dev:       {result.std_ms:>10.2f} ms     ║
║  Min:           {result.min_ms:>10.2f} ms     ║
║  P50:           {result.p50_ms:>10.2f} ms     ║
║  P95:           {result.p95_ms:>10.2f} ms     ║
║  P99:           {result.p99_ms:>10.2f} ms     ║
║  Max:           {result.max_ms:>10.2f} ms     ║
║                                      ║
║  FPS:           {result.fps:>10.1f}        ║
║  Memory:        {result.gpu_memory_mb:>10.1f} MB     ║
║                                      ║
╚══════════════════════════════════════╝
"""

    def print_target_goals(self, baseline: 'BenchmarkResult', target_speedup: float = 50):
        """
        Print optimization target goals

        Args:
            baseline: Baseline BenchmarkResult
            target_speedup: Target speedup multiplier
        """
        target_latency = baseline.latency_ms / target_speedup
        target_fps = baseline.fps * target_speedup

        print("\n" + "🎯" * 35)
        print("🎯 OPTIMIZATION GOALS")
        print("🎯" * 35)
        print(f"\n  Current Performance:")
        print(f"    • Latency: {baseline.latency_ms:.2f} ms")
        print(f"    • FPS: {baseline.fps:.1f}")
        print(f"\n  Target Performance ({target_speedup}x speedup):")
        print(f"    • Latency: <{target_latency:.2f} ms")
        print(f"    • FPS: >{target_fps:.0f}")
        print(f"\n  To achieve this, we'll implement:")
        print(f"    ✓ Fused multi-head attention kernel")
        print(f"    ✓ Fused feed-forward network kernel")
        print(f"    ✓ Optimized instance normalization")
        print(f"    ✓ Memory access optimizations")
        print("\n" + "🎯" * 35)
