"""
StyleForge - Attention Kernel Benchmarking Script

This script provides accurate performance benchmarking of the fused attention
kernel against PyTorch's nn.MultiheadAttention using CUDA events for precise
timing measurements.

Features:
- CUDA event-based timing (nanosecond precision)
- Warmup iterations to avoid cold start effects
- Statistical analysis (mean, std, min, max, median)
- Determinism verification
- GPU memory tracking
- Comprehensive reporting

Usage:
    python kernels/benchmark_attention.py
    python kernels/benchmark_attention.py --config fast
    python kernels/benchmark_attention.py --config comprehensive
"""

import torch
import torch.nn as nn
import time
import gc
import statistics
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.attention_wrapper import FusedAttention
from utils.cuda_build import verify_cuda_installation


class BenchmarkConfig(Enum):
    """Predefined benchmark configurations."""
    FAST = "fast"              # 10 warmup, 50 iterations
    STANDARD = "standard"      # 20 warmup, 100 iterations
    COMPREHENSIVE = "comprehensive"  # 50 warmup, 500 iterations


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    iterations: int
    warmup_iterations: int
    times_ms: List[float] = field(default_factory=list)
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    median_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0

    def compute_statistics(self, batch_size: int, seq_len: int):
        """Compute statistics from raw times."""
        if self.times_ms:
            self.mean_ms = statistics.mean(self.times_ms)
            self.std_ms = statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0
            self.min_ms = min(self.times_ms)
            self.max_ms = max(self.times_ms)
            self.median_ms = statistics.median(self.times_ms)

            # Compute throughput: tokens per second
            # tokens = batch_size * seq_len
            total_tokens = batch_size * seq_len
            avg_time_sec = self.mean_ms / 1000.0
            self.throughput_tokens_per_sec = total_tokens / avg_time_sec


@dataclass
class ComparisonResult:
    """Result comparing CUDA kernel to PyTorch."""
    pytorch_result: BenchmarkResult
    cuda_result: BenchmarkResult
    speedup: float = 0.0
    validation_passed: bool = False
    max_diff: float = 0.0
    mean_diff: float = 0.0
    determinism_passed: bool = False
    gpu_memory_saved_mb: float = 0.0


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{char * width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{text.center(width)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{char * width}{Colors.ENDC}\n")


def print_section(text: str):
    """Print a section header."""
    print(f"\n{Colors.UNDERLINE}{Colors.OKBLUE}{text}{Colors.ENDC}")
    print("-" * len(text))


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}{text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}{text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}{text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}{text}{Colors.ENDC}")


def get_gpu_memory_allocated() -> float:
    """Get current GPU memory allocation in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_gpu_memory_reserved() -> float:
    """Get current GPU memory reservation in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved() / (1024 ** 2)
    return 0.0


def reset_gpu_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


class AttentionBenchmark:
    """
    Benchmark suite for attention kernel vs PyTorch.

    Uses CUDA events for accurate timing and includes validation
    and determinism checks.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        batch_size: int = 1,
        seq_len: int = 256,
        embed_dim: int = 128,
        num_heads: int = 4,
        bias: bool = True
    ):
        """
        Initialize benchmark.

        Args:
            device: CUDA device to use
            batch_size: Batch size for testing
            seq_len: Sequence length for testing
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            bias: Whether to use bias
        """
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for benchmarking")
            device = torch.device('cuda')

        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias

        # Verify configuration
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = embed_dim // num_heads

        # Verify CUDA installation
        is_available, status_msg = verify_cuda_installation()
        if not is_available:
            raise RuntimeError(f"CUDA verification failed: {status_msg}")

        # Get device info
        self.device_name = torch.cuda.get_device_name(0)
        self.compute_capability = torch.cuda.get_device_capability(0)
        self.total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

        print_info(f"Device: {self.device_name}")
        print_info(f"Compute Capability: {self.compute_capability[0]}.{self.compute_capability[1]}")
        print_info(f"Total Memory: {self.total_memory_mb:.0f} MB")

    def create_input(self) -> torch.Tensor:
        """Create random input tensor."""
        return torch.randn(
            self.batch_size,
            self.seq_len,
            self.embed_dim,
            device=self.device,
            dtype=torch.float32
        )

    def create_pytorch_model(self) -> nn.MultiheadAttention:
        """Create PyTorch MultiheadAttention model."""
        model = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            bias=self.bias,
            batch_first=True
        ).to(self.device)
        model.eval()
        return model

    def create_fused_model(self) -> FusedAttention:
        """Create fused attention model."""
        model = FusedAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            bias=self.bias
        ).to(self.device)
        model.eval()
        return model

    def copy_weights(
        self,
        pytorch_model: nn.MultiheadAttention,
        fused_model: FusedAttention
    ):
        """Copy weights from PyTorch to fused model for fair comparison."""
        with torch.no_grad():
            fused_model.w_qkv.copy_(pytorch_model.in_proj_weight)
            # Note: PyTorch's out_proj.weight is [embed_dim, embed_dim]
            # The kernel expects the same layout (row-major)
            fused_model.w_out.copy_(pytorch_model.out_proj.weight)

            if pytorch_model.in_proj_bias is not None and fused_model.bias_qkv is not None:
                fused_model.bias_qkv.copy_(pytorch_model.in_proj_bias)

            if pytorch_model.out_proj.bias is not None and fused_model.bias_out is not None:
                fused_model.bias_out.copy_(pytorch_model.out_proj.bias)

    def validate_correctness(
        self,
        pytorch_model: nn.MultiheadAttention,
        fused_model: FusedAttention,
        x: torch.Tensor
    ) -> Tuple[bool, float, float]:
        """
        Validate that CUDA kernel produces correct results.

        Returns:
            Tuple of (passed, max_diff, mean_diff)
        """
        with torch.no_grad():
            # PyTorch output
            out_pytorch, _ = pytorch_model(x, x, x)

            # Fused output
            out_fused = fused_model(x)

            # Compare
            diff = (out_fused - out_pytorch).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            passed = max_diff < 1e-4

        return passed, max_diff, mean_diff

    def check_determinism(
        self,
        model: nn.Module,
        x: torch.Tensor,
        num_runs: int = 3,
        model_name: str = "model"
    ) -> bool:
        """
        Check that model produces deterministic outputs.

        Returns:
            True if all outputs are identical
        """
        outputs = []
        with torch.no_grad():
            for _ in range(num_runs):
                if isinstance(model, nn.MultiheadAttention):
                    out, _ = model(x, x, x)
                else:
                    out = model(x)
                outputs.append(out.clone())

        # Check all outputs are identical
        for i in range(1, len(outputs)):
            if not torch.allclose(outputs[0], outputs[i], rtol=0, atol=0):
                print_warning(f"{model_name} is not deterministic!")
                return False

        return True

    def benchmark_pytorch(
        self,
        x: torch.Tensor,
        model: nn.Module,
        warmup_iterations: int = 20,
        iterations: int = 100
    ) -> BenchmarkResult:
        """
        Benchmark PyTorch model using CUDA events.

        Args:
            x: Input tensor
            model: PyTorch model
            warmup_iterations: Number of warmup runs
            iterations: Number of timed iterations

        Returns:
            BenchmarkResult with timing statistics
        """
        result = BenchmarkResult(
            name="PyTorch nn.MultiheadAttention",
            iterations=iterations,
            warmup_iterations=warmup_iterations
        )

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(x, x, x)
                torch.cuda.synchronize()

        # Reset memory tracking
        reset_gpu_memory()

        # Timed runs using CUDA events
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(iterations):
                start_event.record()
                _ = model(x, x, x)
                end_event.record()
                torch.cuda.synchronize()

                elapsed_ms = start_event.elapsed_time(end_event)
                result.times_ms.append(elapsed_ms)

        result.compute_statistics(self.batch_size, self.seq_len)

        return result

    def benchmark_fused(
        self,
        x: torch.Tensor,
        model: nn.Module,
        warmup_iterations: int = 20,
        iterations: int = 100
    ) -> BenchmarkResult:
        """
        Benchmark fused attention using CUDA events.

        Args:
            x: Input tensor
            model: Fused attention model
            warmup_iterations: Number of warmup runs
            iterations: Number of timed iterations

        Returns:
            BenchmarkResult with timing statistics
        """
        result = BenchmarkResult(
            name="Fused Attention (CUDA)",
            iterations=iterations,
            warmup_iterations=warmup_iterations
        )

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(x)
                torch.cuda.synchronize()

        # Reset memory tracking
        reset_gpu_memory()

        # Timed runs using CUDA events
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(iterations):
                start_event.record()
                _ = model(x)
                end_event.record()
                torch.cuda.synchronize()

                elapsed_ms = start_event.elapsed_time(end_event)
                result.times_ms.append(elapsed_ms)

        result.compute_statistics(self.batch_size, self.seq_len)

        return result

    def run_comparison(
        self,
        config: BenchmarkConfig = BenchmarkConfig.STANDARD
    ) -> ComparisonResult:
        """
        Run full comparison benchmark.

        Args:
            config: Benchmark configuration

        Returns:
            ComparisonResult with all benchmark data
        """
        # Set iterations based on config
        if config == BenchmarkConfig.FAST:
            warmup, iters = 10, 50
        elif config == BenchmarkConfig.COMPREHENSIVE:
            warmup, iters = 50, 500
        else:  # STANDARD
            warmup, iters = 20, 100

        print_section(f"Running benchmark ({config.value} mode: {warmup} warmup, {iters} iterations)")

        # Create models and input
        print_info("Creating models...")
        x = self.create_input()
        pytorch_model = self.create_pytorch_model()
        fused_model = self.create_fused_model()

        # Copy weights for fair comparison
        self.copy_weights(pytorch_model, fused_model)

        # Step 1: Validate correctness
        print_info("Validating correctness...")
        validation_passed, max_diff, mean_diff = self.validate_correctness(
            pytorch_model, fused_model, x
        )

        if validation_passed:
            print_success(f"  Validation passed! Max diff: {max_diff:.2e}")
        else:
            print_error(f"  Validation FAILED! Max diff: {max_diff:.2e} (tolerance: 1e-4)")

        # Step 2: Check determinism
        print_info("Checking determinism...")
        pytorch_deterministic = self.check_determinism(pytorch_model, x, model_name="PyTorch")
        fused_deterministic = self.check_determinism(fused_model, x, model_name="Fused CUDA")

        if pytorch_deterministic:
            print_success("  PyTorch: Deterministic")
        else:
            print_warning("  PyTorch: Non-deterministic")

        if fused_deterministic:
            print_success("  Fused CUDA: Deterministic")
        else:
            print_warning("  Fused CUDA: Non-deterministic")

        determinism_passed = fused_deterministic

        # Step 3: Benchmark PyTorch
        print_info("Benchmarking PyTorch...")
        pytorch_result = self.benchmark_pytorch(x, pytorch_model, warmup, iters)

        # Step 4: Benchmark Fused
        print_info("Benchmarking Fused CUDA...")
        fused_result = self.benchmark_fused(x, fused_model, warmup, iters)

        # Compute speedup (handle edge cases)
        if fused_result.mean_ms > 0 and not math.isinf(fused_result.mean_ms) and not math.isnan(fused_result.mean_ms):
            speedup = pytorch_result.mean_ms / fused_result.mean_ms
        else:
            speedup = 0.0

        # Create comparison result
        comparison = ComparisonResult(
            pytorch_result=pytorch_result,
            cuda_result=fused_result,
            speedup=speedup,
            validation_passed=validation_passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            determinism_passed=determinism_passed
        )

        return comparison

    def print_results(self, comparison: ComparisonResult):
        """Print comprehensive benchmark results."""
        p = comparison.pytorch_result
        c = comparison.cuda_result

        print_header("Benchmark Results")

        # Configuration
        print(f"{'Configuration:':<30}")
        print(f"  Batch size:     {self.batch_size}")
        print(f"  Sequence length: {self.seq_len}")
        print(f"  Embed dim:      {self.embed_dim}")
        print(f"  Num heads:      {self.num_heads}")
        print(f"  Head dim:       {self.head_dim}")
        print(f"  Bias:           {self.bias}")
        print(f"  Iterations:     {p.iterations}")

        # Validation results
        print_section("Validation")
        if comparison.validation_passed:
            print_success(f"  Correctness:     PASS (max diff: {comparison.max_diff:.2e})")
        else:
            print_error(f"  Correctness:     FAIL (max diff: {comparison.max_diff:.2e})")

        if comparison.determinism_passed:
            print_success(f"  Determinism:     PASS")
        else:
            print_warning(f"  Determinism:     FAIL")

        # Performance table
        print_section("Performance")
        print(f"\n  {'Metric':<25} {'PyTorch':>15} {'Fused CUDA':>15} {'Speedup':>12}")
        print("  " + "-" * 70)

        def format_val(val, fmt=".3f"):
            return f"{val:{fmt}}"

        print(f"  {'Mean (ms)':<25} {format_val(p.mean_ms):>15} {format_val(c.mean_ms):>15} {format_val(comparison.speedup):>12.2f}x")
        print(f"  {'Median (ms)':<25} {format_val(p.median_ms):>15} {format_val(c.median_ms):>15} {format_val(p.median_ms / c.median_ms):>12.2f}x")
        print(f"  {'Std Dev (ms)':<25} {format_val(p.std_ms):>15} {format_val(c.std_ms):>15} {'':>12}")
        print(f"  {'Min (ms)':<25} {format_val(p.min_ms):>15} {format_val(c.min_ms):>15} {format_val(p.min_ms / c.min_ms):>12.2f}x")
        print(f"  {'Max (ms)':<25} {format_val(p.max_ms):>15} {format_val(c.max_ms):>15} {format_val(p.max_ms / c.max_ms):>12.2f}x")

        # Throughput
        print_section("Throughput")
        print(f"  PyTorch:     {p.throughput_tokens_per_sec:,.0f} tokens/sec")
        print(f"  Fused CUDA:  {c.throughput_tokens_per_sec:,.0f} tokens/sec")

        # Verdict
        print_header("Verdict")

        if comparison.validation_passed and comparison.determinism_passed:
            if comparison.speedup >= 2.0:
                print_success(f"  Fused CUDA is {comparison.speedup:.2f}x FASTER than PyTorch")
            elif comparison.speedup >= 1.0:
                print_warning(f"  Fused CUDA is {comparison.speedup:.2f}x faster than PyTorch (marginal improvement)")
            else:
                print_error(f"  Fused CUDA is {1/comparison.speedup:.2f}x SLOWER than PyTorch")
        else:
            print_error(f"  Cannot claim speedup - validation or determinism failed")

        print()


def run_benchmark(
    config: BenchmarkConfig = BenchmarkConfig.STANDARD,
    batch_size: int = 1,
    seq_len: int = 256,
    embed_dim: int = 128,
    num_heads: int = 4,
    bias: bool = True
):
    """
    Run benchmark with specified parameters.

    Args:
        config: Benchmark configuration
        batch_size: Batch size
        seq_len: Sequence length
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        bias: Whether to use bias
    """
    print_header("StyleForge Attention Kernel Benchmark")

    # Check CUDA
    if not torch.cuda.is_available():
        print_error("CUDA is not available. Cannot run benchmark.")
        return

    try:
        benchmark = AttentionBenchmark(
            batch_size=batch_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias
        )

        comparison = benchmark.run_comparison(config)
        benchmark.print_results(comparison)

        # Return for programmatic use
        return comparison

    except Exception as e:
        print_error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_comprehensive_benchmark():
    """Run benchmarks across multiple configurations."""
    print_header("Comprehensive Benchmark Suite")

    if not torch.cuda.is_available():
        print_error("CUDA is not available. Cannot run benchmark.")
        return

    configs = [
        # (batch_size, seq_len, embed_dim, num_heads, bias, name)
        (1, 128, 128, 4, True, "Small (1x128x128)"),
        (1, 256, 128, 4, True, "Medium (1x256x128)"),
        (1, 512, 128, 4, True, "Large sequence (1x512x128)"),
        (2, 256, 128, 4, True, "Batch 2 (2x256x128)"),
        (1, 256, 256, 8, True, "Large embed (1x256x256)"),
    ]

    results = []

    for batch_size, seq_len, embed_dim, num_heads, bias, name in configs:
        print_header(f"Benchmark: {name}")

        try:
            benchmark = AttentionBenchmark(
                batch_size=batch_size,
                seq_len=seq_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
                bias=bias
            )

            # Use fast mode for comprehensive suite
            comparison = benchmark.run_comparison(BenchmarkConfig.FAST)
            comparison.config_name = name
            results.append(comparison)

        except Exception as e:
            print_error(f"Configuration '{name}' failed: {e}")

    # Summary table
    print_header("Comprehensive Results Summary")

    print(f"\n  {'Configuration':<25} {'PyTorch (ms)':>15} {'CUDA (ms)':>15} {'Speedup':>12} {'Valid':>8}")
    print("  " + "-" * 80)

    for r in results:
        valid = "PASS" if r.validation_passed and r.determinism_passed else "FAIL"
        print(f"  {r.config_name:<25} "
              f"{r.pytorch_result.mean_ms:>15.3f} "
              f"{r.cuda_result.mean_ms:>15.3f} "
              f"{r.speedup:>12.2f}x "
              f"{valid:>8}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark fused attention kernel vs PyTorch"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["fast", "standard", "comprehensive"],
        default="standard",
        help="Benchmark configuration"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--no-bias", action="store_true", help="Disable bias")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run comprehensive benchmark suite"
    )

    args = parser.parse_args()

    if args.suite:
        run_comprehensive_benchmark()
    else:
        config_map = {
            "fast": BenchmarkConfig.FAST,
            "standard": BenchmarkConfig.STANDARD,
            "comprehensive": BenchmarkConfig.COMPREHENSIVE,
        }

        run_benchmark(
            config=config_map[args.config],
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            bias=not args.no_bias
        )
