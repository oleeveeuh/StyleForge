"""
Benchmark utilities for StyleForge.

Provides timing, profiling, and performance measurement tools.
"""

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import numpy as np


@dataclass
class TimingResult:
    """Timing statistics for a benchmark run."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    total_iters: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""

    model_name: str
    input_size: tuple
    output_size: tuple
    device: str
    warmup_iters: int
    test_iters: int
    timings: Dict[str, TimingResult]
    memory_mb: float
    throughput_ips: float  # Images per second

    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'device': self.device,
            'warmup_iters': self.warmup_iters,
            'test_iters': self.test_iters,
            'timings': {k: v.to_dict() for k, v in self.timings.items()},
            'memory_mb': self.memory_mb,
            'throughput_ips': self.throughput_ips,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@contextmanager
def timer(name: str, results: Optional[Dict[str, List[float]]] = None):
    """Context manager for timing code blocks.

    Example:
        >>> results = {}
        >>> with timer("operation", results):
        ...     some_function()
        >>> print(results["operation"])  # List of timings
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) * 1000  # ms

    if results is not None:
        if name not in results:
            results[name] = []
        results[name].append(elapsed)


def compute_timing_stats(timings: List[float]) -> TimingResult:
    """Compute statistics from a list of timings in milliseconds."""
    timings_array = np.array(timings)

    return TimingResult(
        name="benchmark",
        mean_ms=float(np.mean(timings_array)),
        std_ms=float(np.std(timings_array)),
        min_ms=float(np.min(timings_array)),
        max_ms=float(np.max(timings_array)),
        median_ms=float(np.median(timings_array)),
        p95_ms=float(np.percentile(timings_array, 95)),
        p99_ms=float(np.percentile(timings_array, 99)),
        total_iters=len(timings),
    )


def benchmark_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    warmup_iters: int = 10,
    test_iters: int = 100,
    name: str = "model",
) -> BenchmarkResult:
    """
    Benchmark a PyTorch model.

    Args:
        model: PyTorch model to benchmark
        input_tensor: Sample input tensor
        warmup_iters: Number of warmup iterations
        test_iters: Number of test iterations
        name: Name for the benchmark

    Returns:
        BenchmarkResult with timing statistics
    """
    device = next(model.parameters()).device
    model.eval()

    timings = defaultdict(list)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Get initial memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated() / 1024**2

    # Benchmark
    with torch.no_grad():
        for _ in range(test_iters):
            with timer("forward", timings):
                output = model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        memory_mb = peak_memory - start_memory
    else:
        memory_mb = 0

    # Compute stats
    forward_stats = compute_timing_stats(timings["forward"])

    # Compute throughput
    throughput_ips = 1000 / forward_stats.mean_ms  # images/sec

    return BenchmarkResult(
        model_name=name,
        input_size=tuple(input_tensor.shape),
        output_size=tuple(output.shape),
        device=str(device),
        warmup_iters=warmup_iters,
        test_iters=test_iters,
        timings={"forward": forward_stats},
        memory_mb=memory_mb,
        throughput_ips=throughput_ips,
    )


def benchmark_layered(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    layers: List[str],
    warmup_iters: int = 5,
    test_iters: int = 50,
) -> Dict[str, TimingResult]:
    """
    Benchmark individual layers/components of a model.

    Args:
        model: PyTorch model
        input_tensor: Sample input
        layers: List of layer attribute names to benchmark
        warmup_iters: Warmup iterations
        test_iters: Test iterations

    Returns:
        Dict mapping layer names to TimingResult
    """
    device = next(model.parameters()).device
    model.eval()

    results = {}

    for layer_name in layers:
        layer = getattr(model, layer_name, None)
        if layer is None:
            print(f"Warning: Layer {layer_name} not found")
            continue

        timings = []

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = layer(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        with torch.no_grad():
            for _ in range(test_iters):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = layer(input_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000)

        results[layer_name] = compute_timing_stats(timings)

    return results


def print_benchmark_results(result: BenchmarkResult) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS: {result.model_name}")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Input size:  {result.input_size}")
    print(f"  Output size: {result.output_size}")
    print(f"  Device:      {result.device}")
    print(f"  Warmup:      {result.warmup_iters} iters")
    print(f"  Test:        {result.test_iters} iters")

    print(f"\nTiming Statistics:")
    for name, stats in result.timings.items():
        print(f"\n  {name}:")
        print(f"    Mean:   {stats.mean_ms:.3f} ms")
        print(f"    Median: {stats.median_ms:.3f} ms")
        print(f"    Std:    {stats.std_ms:.3f} ms")
        print(f"    Min:    {stats.min_ms:.3f} ms")
        print(f"    Max:    {stats.max_ms:.3f} ms")
        print(f"    P95:    {stats.p95_ms:.3f} ms")
        print(f"    P99:    {stats.p99_ms:.3f} ms")

    print(f"\nMemory:")
    print(f"  Peak: {result.memory_mb:.1f} MB")

    print(f"\nThroughput:")
    print(f"  {result.throughput_ips:.2f} images/second")

    # Real-time assessment
    fps = result.throughput_ips
    print(f"\nReal-time capability:")
    if fps >= 30:
        print(f"  ✅ REAL-TIME ({fps:.1f} FPS ≥ 30 FPS)")
    elif fps >= 24:
        print(f"  ✅ VIDEO ({fps:.1f} FPS ≥ 24 FPS)")
    elif fps >= 15:
        print(f"  ⚠️  USABLE ({fps:.1f} FPS - slightly below 30 FPS)")
    else:
        print(f"  ❌ NOT REAL-TIME ({fps:.1f} FPS < 15 FPS)")

    print("=" * 70)


class LayerProfiler:
    """Profile each layer of a model."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.timings = defaultdict(list)
        self.hooks = []

    def _create_hook(self, name: str):
        def hook(module, input, output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.timings[name].append(time.perf_counter())
        return hook

    def start(self) -> None:
        """Register forward hooks for timing."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module only
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._record_time(n)
                )
                self.hooks.append(hook)

    def _record_time(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings[name].append(time.perf_counter())

    def stop(self) -> Dict[str, float]:
        """Remove hooks and return timings."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # Compute durations
        durations = {}
        for name, times in self.timings.items():
            if len(times) >= 2:
                diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
                durations[name] = np.mean(diffs) * 1000  # ms

        return durations

    def print_summary(self) -> None:
        """Print timing summary for each layer."""
        durations = self.stop()

        # Sort by time
        sorted_layers = sorted(durations.items(), key=lambda x: x[1], reverse=True)

        print("\nLayer Timing Breakdown:")
        print("-" * 50)
        for name, duration in sorted_layers:
            print(f"  {name:40s}: {duration:6.3f} ms")
        print("-" * 50)


def compare_benchmarks(
    results: List[BenchmarkResult],
    metric: str = "mean_ms",
) -> None:
    """
    Compare multiple benchmark results.

    Args:
        results: List of BenchmarkResult objects
        metric: Which metric to compare ('mean_ms', 'throughput_ips', etc.)
    """
    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPARISON (by {metric})")
    print("=" * 70)

    # Sort by metric
    if "ms" in metric:
        sorted_results = sorted(results, key=lambda r: getattr(r.timings["forward"], metric))
    else:
        sorted_results = sorted(results, key=lambda r: getattr(r, metric), reverse=True)

    # Print table
    print(f"\n{'Model':<30} {metric:<15} {'Relative':<15}")
    print("-" * 70)

    baseline_value = getattr(sorted_results[0].timings["forward"], metric) if "ms" in metric else getattr(sorted_results[0], metric)

    for result in sorted_results:
        if "ms" in metric:
            value = getattr(result.timings["forward"], metric)
            relative = f"{value/baseline_value:.2f}x"
        else:
            value = getattr(result, metric)
            relative = f"{value/baseline_value:.2f}x"

        print(f"{result.model_name:<30} {value:<15.3f} {relative:<15}")

    print("=" * 70)
