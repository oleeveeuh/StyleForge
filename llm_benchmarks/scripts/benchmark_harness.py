"""
Benchmark harness for measuring kernel performance

Provides consistent timing, warmup, and statistical analysis across all benchmarks.
"""

import torch
import time
import numpy as np
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    name: str
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    iterations: int

    def throughput_per_sec(self) -> float:
        """Compute throughput in operations per second"""
        return 1000.0 / self.mean_ms

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'mean_ms': self.mean_ms,
            'median_ms': self.median_ms,
            'std_ms': self.std_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'p95_ms': self.p95_ms,
            'p99_ms': self.p99_ms,
            'throughput_per_sec': self.throughput_per_sec(),
            'iterations': self.iterations,
        }


class BenchmarkHarness:
    """
    Harness for running and comparing kernel benchmarks

    Features:
    - Automatic warmup
    - CUDA event-based timing
    - Statistical analysis
    - Comparison against baselines
    - JSON export
    """

    def __init__(
        self,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        use_cuda_events: bool = True
    ):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.use_cuda_events = use_cuda_events
        self.results: List[BenchmarkResult] = []

    def benchmark(
        self,
        name: str,
        fn: Callable,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a function with automatic warmup and timing

        Args:
            name: Descriptive name for this benchmark
            fn: Function to benchmark
            *args, **kwargs: Arguments to pass to fn

        Returns:
            BenchmarkResult with timing statistics
        """
        # Warmup
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = fn(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        times_ms = []

        for _ in range(self.benchmark_iterations):
            if self.use_cuda_events and torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                with torch.no_grad():
                    _ = fn(*args, **kwargs)
                end_event.record()

                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                start = time.perf_counter()
                with torch.no_grad():
                    _ = fn(*args, **kwargs)
                end = time.perf_counter()
                elapsed_ms = (end - start) * 1000

            times_ms.append(elapsed_ms)

        # Compute statistics
        times_array = np.array(times_ms)

        result = BenchmarkResult(
            name=name,
            mean_ms=float(np.mean(times_array)),
            median_ms=float(np.median(times_array)),
            std_ms=float(np.std(times_array)),
            min_ms=float(np.min(times_array)),
            max_ms=float(np.max(times_array)),
            p95_ms=float(np.percentile(times_array, 95)),
            p99_ms=float(np.percentile(times_array, 99)),
            iterations=self.benchmark_iterations,
        )

        self.results.append(result)
        return result

    def compare(
        self,
        baseline_name: str,
        baseline_fn: Callable,
        optimized_name: str,
        optimized_fn: Callable,
        *args,
        **kwargs
    ) -> Dict:
        """
        Compare baseline vs optimized implementation

        Returns dictionary with comparison metrics
        """
        print(f"\n{'='*70}")
        print(f"Comparing: {baseline_name} vs {optimized_name}")
        print(f"{'='*70}")

        # Benchmark both
        baseline_result = self.benchmark(baseline_name, baseline_fn, *args, **kwargs)
        optimized_result = self.benchmark(optimized_name, optimized_fn, *args, **kwargs)

        # Compute speedup
        speedup = baseline_result.mean_ms / optimized_result.mean_ms

        # Print results
        print(f"\n{baseline_name}:")
        print(f"  Mean:   {baseline_result.mean_ms:.3f} ms")
        print(f"  Median: {baseline_result.median_ms:.3f} ms")
        print(f"  Std:    {baseline_result.std_ms:.3f} ms")

        print(f"\n{optimized_name}:")
        print(f"  Mean:   {optimized_result.mean_ms:.3f} ms")
        print(f"  Median: {optimized_result.median_ms:.3f} ms")
        print(f"  Std:    {optimized_result.std_ms:.3f} ms")

        print(f"\nSpeedup: {speedup:.2f}x")

        if speedup >= 2.0:
            print("Excellent speedup!")
        elif speedup >= 1.5:
            print("Good speedup")
        elif speedup >= 1.2:
            print("Modest speedup")
        elif speedup >= 1.0:
            print("~ Marginal improvement")
        else:
            print(" Slower than baseline")

        return {
            'baseline': baseline_result.to_dict(),
            'optimized': optimized_result.to_dict(),
            'speedup': speedup,
        }

    def save_results(self, filepath: str):
        """Save all benchmark results to JSON file"""
        data = {
            'results': [r.to_dict() for r in self.results],
            'config': {
                'warmup_iterations': self.warmup_iterations,
                'benchmark_iterations': self.benchmark_iterations,
            }
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n Results saved to: {filepath}")

    def print_summary(self):
        """Print summary table of all benchmarks"""
        if not self.results:
            print("No benchmark results yet")
            return

        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"{'Name':<40} {'Mean (ms)':<12} {'Speedup':<10}")
        print("-"*70)

        baseline_time = self.results[0].mean_ms if self.results else 1.0

        for result in self.results:
            speedup = baseline_time / result.mean_ms
            print(f"{result.name:<40} {result.mean_ms:<12.3f} {speedup:<10.2f}x")
