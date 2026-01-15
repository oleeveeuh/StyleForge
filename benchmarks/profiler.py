"""
StyleForge - Performance Profiler

Accurate GPU benchmarking with CUDA events for measuring
model latency, throughput, and memory usage.
"""

import time
import numpy as np
import torch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class BenchmarkResult:
    """Store benchmark metrics

    Attributes:
        name: Benchmark/model name
        latency_ms: Mean latency in milliseconds
        std_ms: Standard deviation of latency
        min_ms: Minimum latency observed
        max_ms: Maximum latency observed
        p50_ms: 50th percentile (median) latency
        p95_ms: 95th percentile latency
        p99_ms: 99th percentile latency
        fps: Frames per second
        throughput_imgs_per_sec: Images processed per second
        gpu_memory_mb: GPU memory used in MB
    """
    name: str
    latency_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    throughput_imgs_per_sec: float
    gpu_memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'latency_ms': round(self.latency_ms, 2),
            'std_ms': round(self.std_ms, 2),
            'min_ms': round(self.min_ms, 2),
            'max_ms': round(self.max_ms, 2),
            'p50_ms': round(self.p50_ms, 2),
            'p95_ms': round(self.p95_ms, 2),
            'p99_ms': round(self.p99_ms, 2),
            'fps': round(self.fps, 1),
            'throughput': round(self.throughput_imgs_per_sec, 1),
            'memory_mb': round(self.gpu_memory_mb, 1)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create BenchmarkResult from dictionary"""
        return cls(
            name=data['name'],
            latency_ms=data['latency_ms'],
            std_ms=data['std_ms'],
            min_ms=data['min_ms'],
            max_ms=data['max_ms'],
            p50_ms=data['p50_ms'],
            p95_ms=data['p95_ms'],
            p99_ms=data['p99_ms'],
            fps=data['fps'],
            throughput_imgs_per_sec=data['throughput'],
            gpu_memory_mb=data['memory_mb']
        )


class PerformanceProfiler:
    """
    Accurate GPU benchmarking with CUDA events

    Uses torch.cuda.Event for precise GPU kernel timing measurements.
    Includes warmup iterations to ensure stable performance measurements.
    """

    def __init__(self, warmup_iters: int = 10, bench_iters: int = 100):
        """
        Args:
            warmup_iters: Number of warmup iterations before benchmarking
            bench_iters: Number of benchmark iterations
        """
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters

    def benchmark(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        name: str,
        show_progress: bool = True
    ) -> tuple[BenchmarkResult, np.ndarray]:
        """
        Benchmark model with proper GPU synchronization

        Args:
            model: PyTorch model (already on GPU)
            input_tensor: Input tensor (already on GPU)
            name: Benchmark name
            show_progress: Whether to print progress

        Returns:
            Tuple of (BenchmarkResult, raw_times_array)
        """
        model.eval()

        # Warmup
        if show_progress:
            print(f"  Warming up '{name}' ({self.warmup_iters} iterations)...")

        with torch.no_grad():
            for _ in range(self.warmup_iters):
                _ = model(input_tensor)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Benchmark
        if show_progress:
            print(f"  Benchmarking '{name}' ({self.bench_iters} iterations)...")

        times_ms = []

        with torch.no_grad():
            for i in range(self.bench_iters):
                # CUDA events for accurate timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # Measure
                start_event.record()
                _ = model(input_tensor)
                end_event.record()

                # Synchronize and get time
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                times_ms.append(elapsed_ms)

                # Progress
                if show_progress and (i + 1) % 25 == 0:
                    print(f"    Progress: {i+1}/{self.bench_iters}")

        # Calculate statistics
        times_ms = np.array(times_ms)
        latency_ms = np.mean(times_ms)
        std_ms = np.std(times_ms)
        min_ms = np.min(times_ms)
        max_ms = np.max(times_ms)
        p50_ms = np.percentile(times_ms, 50)
        p95_ms = np.percentile(times_ms, 95)
        p99_ms = np.percentile(times_ms, 99)

        fps = 1000.0 / latency_ms
        throughput = fps * input_tensor.size(0)  # Account for batch size

        # GPU memory
        memory_mb = torch.cuda.max_memory_allocated() / 1e6

        result = BenchmarkResult(
            name=name,
            latency_ms=latency_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            fps=fps,
            throughput_imgs_per_sec=throughput,
            gpu_memory_mb=memory_mb
        )

        return result, times_ms

    def compare(
        self,
        results: list[BenchmarkResult],
        baseline_name: str = "PyTorch Baseline"
    ) -> Dict[str, Any]:
        """
        Compare multiple benchmark results

        Args:
            results: List of BenchmarkResult objects
            baseline_name: Name of baseline result

        Returns:
            Dictionary with comparison metrics
        """
        baseline = next((r for r in results if r.name == baseline_name), None)
        if baseline is None:
            baseline = results[0]

        comparison = {
            'baseline': baseline.to_dict(),
            'speedups': []
        }

        for result in results:
            if result.name != baseline.name:
                speedup = baseline.latency_ms / result.latency_ms
                comparison['speedups'].append({
                    'name': result.name,
                    'speedup': round(speedup, 2),
                    'latency_ms': round(result.latency_ms, 2),
                    'baseline_ms': round(baseline.latency_ms, 2)
                })

        return comparison

    @staticmethod
    def print_result(result: BenchmarkResult):
        """Pretty print benchmark results"""
        print("\n" + "=" * 70)
        print(f"  {result.name} - Benchmark Results")
        print("=" * 70)
        print(f"  Latency (mean):  {result.latency_ms:>10.2f} ms  (± {result.std_ms:.2f} ms)")
        print(f"  Latency (p50):   {result.p50_ms:>10.2f} ms")
        print(f"  Latency (p95):   {result.p95_ms:>10.2f} ms")
        print(f"  Latency (p99):   {result.p99_ms:>10.2f} ms")
        print(f"  Range:           {result.min_ms:>10.2f} ms - {result.max_ms:.2f} ms")
        print(f"  Throughput:      {result.fps:>10.1f} FPS")
        print(f"  GPU Memory:      {result.gpu_memory_mb:>10.1f} MB")
        print("=" * 70 + "\n")

    @staticmethod
    def print_comparison(results: list[BenchmarkResult], baseline_name: str = "PyTorch Baseline"):
        """Print comparison table of multiple results"""
        baseline = next((r for r in results if r.name == baseline_name), None)
        if baseline is None and results:
            baseline = results[0]

        if not baseline:
            print("No results to compare")
            return

        print("\n" + "=" * 80)
        print("  PERFORMANCE COMPARISON")
        print("=" * 80)
        print(f"{'Model':<25} {'Latency (ms)':>15} {'FPS':>10} {'Speedup':>10}")
        print("-" * 80)

        for result in results:
            speedup = baseline.latency_ms / result.latency_ms if result != baseline else 1.0
            print(f"{result.name:<25} {result.latency_ms:>15.2f} {result.fps:>10.1f} {speedup:>10.2f}x")

        print("=" * 80 + "\n")


def save_results(
    results: Dict[str, Any],
    filepath: str | Path,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save benchmark results to JSON file

    Args:
        results: Results dictionary from profiler
        filepath: Path to save JSON file
        metadata: Optional metadata (GPU info, timestamp, etc.)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    output = results.copy()
    if metadata:
        output['metadata'] = metadata

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def load_results(filepath: str | Path) -> Dict[str, Any]:
    """
    Load benchmark results from JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with benchmark results
    """
    with open(filepath, 'r') as f:
        return json.load(f)
