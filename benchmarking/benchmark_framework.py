"""
StyleForge - Core Benchmarking Framework

Provides automated benchmarking infrastructure for CUDA kernels including:
- Warmup and timing with CUDA events
- Statistical analysis (mean, median, percentiles)
- Correctness validation
- Comparison against baselines
- JSON export for reports
"""

import torch
import time
import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Callable, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    batch_size: int
    channels: int
    height: int
    width: int
    iterations: int = 100
    warmup_iterations: int = 20
    dtype: torch.dtype = torch.float32

    def get_input_shape(self) -> Tuple[int, ...]:
        """Get input tensor shape."""
        return (self.batch_size, self.channels, self.height, self.width)

    def get_input_size_bytes(self) -> int:
        """Get input tensor size in bytes."""
        numel = np.prod(self.get_input_shape())
        return numel * self._get_dtype_bytes()

    def _get_dtype_bytes(self) -> int:
        """Get bytes per element for dtype."""
        if self.dtype == torch.float32:
            return 4
        elif self.dtype == torch.float16:
            return 2
        elif self.dtype == torch.bfloat16:
            return 2
        else:
            return 4


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    throughput_fps: float
    iterations: int
    timestamp: str
    raw_times_ms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['raw_times_ms'] = self.raw_times_ms  # Keep as list
        return d


@dataclass
class ComparisonResult:
    """Results from comparing baseline vs optimized."""
    config: str
    baseline: BenchmarkResult
    optimized: BenchmarkResult
    speedup: float
    max_error: float
    passed_validation: bool
    memory_transfer_reduction: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': self.config,
            'baseline': self.baseline.to_dict(),
            'optimized': self.optimized.to_dict(),
            'speedup': self.speedup,
            'max_error': self.max_error,
            'passed_validation': self.passed_validation,
            'memory_transfer_reduction': self.memory_transfer_reduction,
            'timestamp': self.timestamp,
        }


class BenchmarkFramework:
    """
    Automated benchmarking framework for CUDA kernels.

    Features:
    - Automated warmup and timing with CUDA events
    - Statistical analysis (mean, median, std, percentiles)
    - Correctness validation
    - Comparison against baselines
    - JSON export for reports
    - GPU utilization tracking
    """

    def __init__(
        self,
        use_cuda_events: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            use_cuda_events: Use CUDA events for timing (more accurate)
            device: Device to run benchmarks on (default: cuda if available)
        """
        self.use_cuda_events = use_cuda_events
        self.device = device or self._get_default_device()
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[ComparisonResult] = []

    @staticmethod
    def _get_default_device() -> torch.device:
        """Get default device (CUDA if available, else CPU)."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def create_input_tensor(self, config: BenchmarkConfig) -> torch.Tensor:
        """Create input tensor from configuration."""
        shape = config.get_input_shape()
        return torch.randn(shape, dtype=config.dtype, device=self.device)

    def benchmark_function(
        self,
        func: Callable,
        config: BenchmarkConfig,
        input_tensor: Optional[torch.Tensor] = None,
        description: Optional[str] = None
    ) -> BenchmarkResult:
        """
        Benchmark a function with given configuration.

        Args:
            func: Function to benchmark (takes tensor, returns tensor)
            config: Benchmark configuration
            input_tensor: Pre-created input tensor (created from config if None)
            description: Optional description for logging

        Returns:
            BenchmarkResult with statistics
        """
        if input_tensor is None:
            input_tensor = self.create_input_tensor(config)

        name = description or config.name

        # Warmup
        for _ in range(config.warmup_iterations):
            with torch.no_grad():
                _ = func(input_tensor)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        times_ms = []

        for _ in range(config.iterations):
            if self.use_cuda_events and self.device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                with torch.no_grad():
                    output = func(input_tensor)
                end_event.record()

                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                start = time.perf_counter()
                with torch.no_grad():
                    output = func(input_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                elapsed_ms = (end - start) * 1000

            times_ms.append(elapsed_ms)

        # Compute statistics
        times_array = np.array(times_ms)

        result = BenchmarkResult(
            config_name=name,
            mean_ms=float(np.mean(times_array)),
            median_ms=float(np.median(times_array)),
            std_ms=float(np.std(times_array)),
            min_ms=float(np.min(times_array)),
            max_ms=float(np.max(times_array)),
            p95_ms=float(np.percentile(times_array, 95)),
            p99_ms=float(np.percentile(times_array, 99)),
            throughput_fps=1000.0 / float(np.mean(times_array)),
            iterations=config.iterations,
            timestamp=datetime.now().isoformat(),
            raw_times_ms=times_ms.tolist()
        )

        self.results.append(result)
        return result

    def compare(
        self,
        baseline_func: Callable,
        optimized_func: Callable,
        config: BenchmarkConfig,
        input_tensor: Optional[torch.Tensor] = None,
        validate: bool = True,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        verbose: bool = True
    ) -> Optional[ComparisonResult]:
        """
        Compare baseline vs optimized implementation.

        Args:
            baseline_func: Baseline function (e.g., PyTorch)
            optimized_func: Optimized function (e.g., CUDA kernel)
            config: Benchmark configuration
            input_tensor: Pre-created input tensor
            validate: Run correctness validation
            rtol: Relative tolerance for validation
            atol: Absolute tolerance for validation
            verbose: Print progress

        Returns:
            ComparisonResult or None if validation failed
        """
        if input_tensor is None:
            input_tensor = self.create_input_tensor(config)

        if verbose:
            print(f"\n{'='*70}")
            print(f"Benchmark: {config.name}")
            print(f"Shape: {config.get_input_shape()}")
            print(f"{'='*70}")

        # Validate correctness if requested
        max_error = 0.0
        passed_validation = True

        if validate:
            is_correct, max_error = self.validate_correctness(
                baseline_func, optimized_func, input_tensor, rtol, atol
            )

            if verbose:
                if is_correct:
                    print(f"  Validation: ✓ PASSED (max error: {max_error:.2e})")
                else:
                    print(f"  Validation: ✗ FAILED (max error: {max_error:.2e})")

            passed_validation = is_correct

            if not is_correct:
                return None

        # Benchmark baseline
        baseline_result = self.benchmark_function(
            baseline_func, config, input_tensor,
            description=f"{config.name} (Baseline)"
        )

        if verbose:
            print(f"  Baseline:  {baseline_result.mean_ms:.4f} ± {baseline_result.std_ms:.4f} ms")

        # Benchmark optimized
        optimized_result = self.benchmark_function(
            optimized_func, config, input_tensor,
            description=f"{config.name} (Optimized)"
        )

        if verbose:
            print(f"  Optimized: {optimized_result.mean_ms:.4f} ± {optimized_result.std_ms:.4f} ms")

        # Compute speedup
        speedup = baseline_result.mean_ms / optimized_result.mean_ms
        memory_reduction = self._estimate_memory_reduction(speedup)

        if verbose:
            print(f"  Speedup:   {speedup:.2f}x")
            print(f"  Status:    {self._get_speedup_status(speedup)}")

        comparison = ComparisonResult(
            config=config.name,
            baseline=baseline_result,
            optimized=optimized_result,
            speedup=speedup,
            max_error=max_error,
            passed_validation=passed_validation,
            memory_transfer_reduction=memory_reduction,
            timestamp=datetime.now().isoformat()
        )

        self.comparisons.append(comparison)
        return comparison

    @staticmethod
    def validate_correctness(
        baseline_func: Callable,
        optimized_func: Callable,
        input_tensor: torch.Tensor,
        rtol: float = 1e-4,
        atol: float = 1e-6
    ) -> Tuple[bool, float]:
        """
        Validate that optimized function produces same results as baseline.

        Returns:
            (is_correct, max_error)
        """
        with torch.no_grad():
            baseline_out = baseline_func(input_tensor)
            optimized_out = optimized_func(input_tensor)

        max_error = (baseline_out - optimized_out).abs().max().item()

        is_correct = torch.allclose(
            baseline_out, optimized_out,
            rtol=rtol, atol=atol
        )

        return is_correct, max_error

    @staticmethod
    def _estimate_memory_reduction(speedup: float) -> str:
        """
        Estimate memory bandwidth reduction from speedup.

        For fused kernels, speedup often comes from reduced memory transfers.
        """
        if speedup >= 5.0:
            return "60-80%"
        elif speedup >= 3.0:
            return "40-60%"
        elif speedup >= 2.0:
            return "20-40%"
        else:
            return "<20%"

    @staticmethod
    def _get_speedup_status(speedup: float) -> str:
        """Get status message for speedup value."""
        if speedup >= 5.0:
            return "✓✓✓ Excellent!"
        elif speedup >= 3.0:
            return "✓✓ Good"
        elif speedup >= 2.0:
            return "✓ Modest"
        elif speedup >= 1.0:
            return "⚠️ Minimal"
        else:
            return "✗ Slower"

    def run_suite(
        self,
        baseline_func: Callable,
        optimized_func: Callable,
        configs: List[BenchmarkConfig],
        validate_first: bool = True,
        verbose: bool = True
    ) -> List[ComparisonResult]:
        """
        Run benchmark suite across multiple configurations.

        Args:
            baseline_func: Baseline function
            optimized_func: Optimized function
            configs: List of benchmark configurations
            validate_first: Validate correctness on first config only
            verbose: Print progress

        Returns:
            List of ComparisonResults
        """
        results = []

        for i, config in enumerate(configs):
            # Only validate on first run if validate_first is True
            validate = (i == 0) if validate_first else True

            result = self.compare(
                baseline_func=baseline_func,
                optimized_func=optimized_func,
                config=config,
                validate=validate,
                verbose=verbose
            )

            if result is not None:
                results.append(result)

        return results

    def save_results(self, filepath: str):
        """Save all benchmark results to JSON."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'results': [r.to_dict() for r in self.results],
            'comparisons': [c.to_dict() for c in self.comparisons],
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        if self._should_print():
            print(f"\n✅ Results saved to: {filepath}")

    def load_results(self, filepath: str) -> Dict:
        """Load benchmark results from JSON."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def print_summary(self):
        """Print summary of all benchmarks."""
        if not self.comparisons:
            print("No comparison results yet")
            return

        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        print(f"\n{'Config':<25} {'Baseline':<12} {'Optimized':<12} {'Speedup':<10}")
        print("-" * 70)

        for comp in self.comparisons:
            baseline_ms = comp.baseline.mean_ms
            optimized_ms = comp.optimized.mean_ms
            speedup = comp.speedup
            status = "✓" if comp.passed_validation else "✗"

            print(
                f"{comp.config:<24} {baseline_ms:<12.4f} "
                f"{optimized_ms:<12.4f} {speedup:<10.2f}x {status}"
            )

        # Calculate averages
        if self.comparisons:
            avg_speedup = np.mean([c.speedup for c in self.comparisons])
            print("-" * 70)
            print(f"{'Average Speedup':<36} {avg_speedup:<10.2f}x")

        print()

    def _should_print(self) -> bool:
        """Check if we should print messages (not in quiet mode)."""
        return True  # Can be made configurable

    def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get current GPU information."""
        if self.device.type != 'cuda':
            return None

        props = torch.cuda.get_device_properties(self.device)

        return {
            'name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'total_memory_gb': props.total_memory / 1e9,
            'multi_processor_count': props.multi_processor_count,
            'cuda_version': torch.version.cuda,
            'torch_version': torch.__version__,
        }
