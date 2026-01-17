"""
StyleForge - Visualization Module

Generate charts and graphs from benchmark results.

Requires matplotlib (optional dependency).
"""

import warnings
from typing import List, Dict, Optional
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class BenchmarkVisualizer:
    """Generate visualizations from benchmark results."""

    def __init__(self, style: str = 'default'):
        """
        Args:
            style: Matplotlib style to use
        """
        if not HAS_MATPLOTLIB:
            warnings.warn(
                "matplotlib not available. Charts will be skipped. "
                "Install with: pip install matplotlib"
            )
            return

        if style != 'default':
            plt.style.use(style)

        self.colors = {
            'pytorch': '#1f77b4',
            'cuda': '#2ca02c',
            'speedup': '#ff7f0e',
            'background': '#f5f5f5',
        }

    def plot_speedup_comparison(
        self,
        results: List[Dict],
        output_path: str,
        figsize: tuple = (12, 6)
    ):
        """
        Plot speedup comparison across configurations.

        Args:
            results: List of benchmark comparison results
            output_path: Path to save figure
            figsize: Figure size (width, height)
        """
        if not HAS_MATPLOTLIB:
            return

        configs = []
        speedups = []

        for result in results:
            if result is None:
                continue
            configs.append(result.get('config', ''))
            speedups.append(result.get('speedup', 0))

        fig, ax = plt.subplots(figsize=figsize)

        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, speedups, color=self.colors['speedup'], alpha=0.8)

        # Add value labels on bars
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{speedup:.1f}x',
                   ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Speedup (x)')
        ax.set_title('CUDA Kernel Speedup vs PyTorch Baseline')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')
        ax.axhline(y=3.0, color='green', linestyle=':', alpha=0.3, label='3x target')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Speedup chart saved to: {output_path}")

    def plot_latency_comparison(
        self,
        results: List[Dict],
        output_path: str,
        figsize: tuple = (12, 6)
    ):
        """
        Plot latency comparison (bar chart with side-by-side bars).

        Args:
            results: List of benchmark comparison results
            output_path: Path to save figure
            figsize: Figure size (width, height)
        """
        if not HAS_MATPLOTLIB:
            return

        configs = []
        baseline_times = []
        optimized_times = []

        for result in results:
            if result is None:
                continue
            configs.append(result.get('config', ''))
            baseline = result.get('baseline', {})
            optimized = result.get('optimized', {})
            baseline_times.append(baseline.get('mean_ms', 0))
            optimized_times.append(optimized.get('mean_ms', 0))

        fig, ax = plt.subplots(figsize=figsize)

        x_pos = np.arange(len(configs))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, baseline_times, width,
                       label='PyTorch', color=self.colors['pytorch'], alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, optimized_times, width,
                       label='CUDA', color=self.colors['cuda'], alpha=0.8)

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Comparison: PyTorch vs CUDA')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Latency chart saved to: {output_path}")

    def plot_latency_distribution(
        self,
        results: List[Dict],
        output_path: str,
        figsize: tuple = (12, 6)
    ):
        """
        Plot latency distribution (box plot).

        Args:
            results: List of benchmark comparison results with raw timing data
            output_path: Path to save figure
            figsize: Figure size (width, height)
        """
        if not HAS_MATPLOTLIB:
            return

        # This requires raw timing data
        configs = []
        baseline_data = []
        optimized_data = []

        for result in results:
            if result is None:
                continue
            baseline = result.get('baseline', {})
            optimized = result.get('optimized', {})

            raw_baseline = baseline.get('raw_times_ms', [])
            raw_optimized = optimized.get('raw_times_ms', [])

            if not raw_baseline or not raw_optimized:
                continue

            configs.append(result.get('config', ''))
            baseline_data.append(raw_baseline)
            optimized_data.append(raw_optimized)

        if not configs:
            print("⚠️  No raw timing data available for distribution plot")
            return

        fig, ax = plt.subplots(figsize=figsize)

        x_pos = np.arange(len(configs))
        width = 0.35

        bp1 = ax.boxplot(baseline_data, positions=x_pos - width/2, widths=width,
                         patch_artist=True, boxprops=dict(facecolor=self.colors['pytorch'], alpha=0.7))
        bp2 = ax.boxplot(optimized_data, positions=x_pos + width/2, widths=width,
                         patch_artist=True, boxprops=dict(facecolor=self.colors['cuda'], alpha=0.7))

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Distribution: PyTorch vs CUDA')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['PyTorch', 'CUDA'])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Distribution chart saved to: {output_path}")

    def plot_throughput_comparison(
        self,
        results: List[Dict],
        output_path: str,
        figsize: tuple = (12, 6)
    ):
        """
        Plot throughput (FPS) comparison.

        Args:
            results: List of benchmark comparison results
            output_path: Path to save figure
            figsize: Figure size (width, height)
        """
        if not HAS_MATPLOTLIB:
            return

        configs = []
        baseline_fps = []
        optimized_fps = []

        for result in results:
            if result is None:
                continue
            configs.append(result.get('config', ''))
            baseline = result.get('baseline', {})
            optimized = result.get('optimized', {})
            baseline_fps.append(baseline.get('throughput_fps', 0))
            optimized_fps.append(optimized.get('throughput_fps', 0))

        fig, ax = plt.subplots(figsize=figsize)

        x_pos = np.arange(len(configs))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, baseline_fps, width,
                       label='PyTorch', color=self.colors['pytorch'], alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, optimized_fps, width,
                       label='CUDA', color=self.colors['cuda'], alpha=0.8)

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Throughput (FPS)')
        ax.set_title('Throughput Comparison: PyTorch vs CUDA')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Throughput chart saved to: {output_path}")

    def generate_all_charts(
        self,
        results: List[Dict],
        output_dir: str
    ):
        """
        Generate all chart types.

        Args:
            results: List of benchmark results
            output_dir: Directory to save charts
        """
        if not HAS_MATPLOTLIB:
            print("⚠️  Charts skipped (matplotlib not available)")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.plot_speedup_comparison(results, output_dir / "speedup_comparison.png")
        self.plot_latency_comparison(results, output_dir / "latency_comparison.png")
        self.plot_latency_distribution(results, output_dir / "latency_distribution.png")
        self.plot_throughput_comparison(results, output_dir / "throughput_comparison.png")

        print(f"✅ All charts saved to: {output_dir}")
