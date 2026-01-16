"""
StyleForge - Comprehensive Fused Attention Test Suite

This test suite validates the CUDA fused attention kernel against
PyTorch's nn.MultiheadAttention across various configurations.

Test coverage:
- Different batch sizes
- Different sequence lengths
- Different embedding dimensions
- Different numbers of attention heads
- With and without bias
- Performance benchmarking
"""

import torch
import torch.nn as nn
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from itertools import product

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.attention_wrapper import FusedAttention
from utils.cuda_build import verify_cuda_installation


@dataclass
class TestConfig:
    """Configuration for a single test case."""
    batch_size: int
    seq_len: int
    embed_dim: int
    num_heads: int
    bias: bool
    dtype: torch.dtype = torch.float32


@dataclass
class TestResult:
    """Result of a single test case."""
    config: TestConfig
    passed: bool
    max_diff: float
    mean_diff: float
    pytorch_time_ms: float
    cuda_time_ms: float
    speedup: float
    error_message: Optional[str] = None


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


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n{Colors.UNDERLINE}{Colors.OKBLUE}{text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}  {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}  {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}  {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}  {text}{Colors.ENDC}")


class AttentionTestSuite:
    """
    Comprehensive test suite for fused attention kernel.

    Tests the CUDA kernel against PyTorch's nn.MultiheadAttention
    across various configurations and measures performance.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the test suite.

        Args:
            device: Device to run tests on. If None, uses CUDA if available.
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                raise RuntimeError(
                    "CUDA is not available. "
                    "The fused attention kernel requires CUDA."
                )

        self.device = device
        self.results: List[TestResult] = []

        # Verify CUDA installation
        is_available, status_msg = verify_cuda_installation()
        if not is_available:
            raise RuntimeError(f"CUDA verification failed: {status_msg}")

        print_info(f"Using device: {self.device}")
        print_info(f"Status: {status_msg}")

    def create_test_configs(self) -> List[TestConfig]:
        """
        Create test configurations covering various scenarios.

        Returns:
            List of TestConfig objects
        """
        configs = []

        # Test 1: Different batch sizes
        for batch_size in [1, 2, 4, 8]:
            configs.append(TestConfig(
                batch_size=batch_size,
                seq_len=128,
                embed_dim=128,
                num_heads=4,
                bias=True
            ))

        # Test 2: Different sequence lengths
        for seq_len in [64, 128, 256, 512]:
            configs.append(TestConfig(
                batch_size=2,
                seq_len=seq_len,
                embed_dim=128,
                num_heads=4,
                bias=True
            ))

        # Test 3: Different embedding dimensions
        for embed_dim in [128, 256, 512]:
            configs.append(TestConfig(
                batch_size=2,
                seq_len=128,
                embed_dim=embed_dim,
                num_heads=4,
                bias=True
            ))

        # Test 4: Different numbers of heads
        for num_heads in [1, 2, 4, 8]:
            configs.append(TestConfig(
                batch_size=2,
                seq_len=128,
                embed_dim=128,
                num_heads=num_heads,
                bias=True
            ))

        # Test 5: With and without bias
        for bias in [True, False]:
            configs.append(TestConfig(
                batch_size=2,
                seq_len=128,
                embed_dim=128,
                num_heads=4,
                bias=bias
            ))

        # Test 6: Edge cases
        # Minimal configuration
        configs.append(TestConfig(
            batch_size=1,
            seq_len=32,
            embed_dim=64,
            num_heads=2,
            bias=True
        ))

        # Large batch, small sequence
        configs.append(TestConfig(
            batch_size=16,
            seq_len=64,
            embed_dim=128,
            num_heads=4,
            bias=True
        ))

        # Small batch, large sequence
        configs.append(TestConfig(
            batch_size=1,
            seq_len=512,
            embed_dim=256,
            num_heads=8,
            bias=True
        ))

        return configs

    def copy_weights_from_pytorch(
        self,
        pytorch_attn: nn.MultiheadAttention,
        fused_attn: FusedAttention
    ):
        """
        Copy weights from PyTorch attention to fused attention.

        This ensures both models have identical weights for fair comparison.

        Args:
            pytorch_attn: PyTorch MultiheadAttention module
            fused_attn: FusedAttention module
        """
        with torch.no_grad():
            # PyTorch's in_proj_weight layout: [Q_weights; K_weights; V_weights]
            # This is the same layout our fused attention expects
            fused_attn.w_qkv.copy_(pytorch_attn.in_proj_weight)

            # Output projection weight
            fused_attn.w_out.copy_(pytorch_attn.out_proj.weight.T)

            # Copy bias if present
            if pytorch_attn.in_proj_bias is not None and fused_attn.bias_qkv is not None:
                fused_attn.bias_qkv.copy_(pytorch_attn.in_proj_bias)

            if pytorch_attn.out_proj.bias is not None and fused_attn.bias_out is not None:
                fused_attn.bias_out.copy_(pytorch_attn.out_proj.bias)

    def run_single_test(
        self,
        config: TestConfig,
        num_warmup: int = 3,
        num_iterations: int = 10
    ) -> TestResult:
        """
        Run a single test case.

        Args:
            config: Test configuration
            num_warmup: Number of warmup iterations (not timed)
            num_iterations: Number of timed iterations

        Returns:
            TestResult object
        """
        try:
            # Validate configuration
            if config.embed_dim % config.num_heads != 0:
                return TestResult(
                    config=config,
                    passed=False,
                    max_diff=float('inf'),
                    mean_diff=float('inf'),
                    pytorch_time_ms=0,
                    cuda_time_ms=0,
                    speedup=0,
                    error_message=f"embed_dim ({config.embed_dim}) must be divisible by num_heads ({config.num_heads})"
                )

            # Create input
            x = torch.randn(
                config.batch_size,
                config.seq_len,
                config.embed_dim,
                device=self.device,
                dtype=config.dtype
            )

            # Initialize PyTorch attention
            pytorch_attn = nn.MultiheadAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                bias=config.bias,
                batch_first=True
            ).to(self.device)

            # Initialize fused attention
            fused_attn = FusedAttention(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                bias=config.bias
            ).to(self.device)

            # Copy weights from PyTorch to fused for fair comparison
            self.copy_weights_from_pytorch(pytorch_attn, fused_attn)

            # =========================================================================
            # PyTorch Baseline - Warmup
            # =========================================================================
            with torch.no_grad():
                for _ in range(num_warmup):
                    out_pytorch, _ = pytorch_attn(x, x, x)
                    torch.cuda.synchronize()

            # =========================================================================
            # PyTorch Baseline - Timed Runs
            # =========================================================================
            with torch.no_grad():
                start_time = time.perf_counter()
                for _ in range(num_iterations):
                    out_pytorch, _ = pytorch_attn(x, x, x)
                torch.cuda.synchronize()
                pytorch_time = (time.perf_counter() - start_time) * 1000 / num_iterations

            # =========================================================================
            # CUDA Fused Attention - Warmup
            # =========================================================================
            with torch.no_grad():
                for _ in range(num_warmup):
                    out_fused = fused_attn(x)
                    torch.cuda.synchronize()

            # =========================================================================
            # CUDA Fused Attention - Timed Runs
            # =========================================================================
            with torch.no_grad():
                start_time = time.perf_counter()
                for _ in range(num_iterations):
                    out_fused = fused_attn(x)
                torch.cuda.synchronize()
                cuda_time = (time.perf_counter() - start_time) * 1000 / num_iterations

            # =========================================================================
            # Compare Outputs
            # =========================================================================
            diff = (out_fused - out_pytorch).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Check tolerance
            tolerance = 1e-4
            passed = max_diff < tolerance

            # Calculate speedup
            speedup = pytorch_time / cuda_time if cuda_time > 0 else 0

            return TestResult(
                config=config,
                passed=passed,
                max_diff=max_diff,
                mean_diff=mean_diff,
                pytorch_time_ms=pytorch_time,
                cuda_time_ms=cuda_time,
                speedup=speedup,
                error_message=None
            )

        except Exception as e:
            return TestResult(
                config=config,
                passed=False,
                max_diff=float('inf'),
                mean_diff=float('inf'),
                pytorch_time_ms=0,
                cuda_time_ms=0,
                speedup=0,
                error_message=f"{type(e).__name__}: {str(e)}"
            )

    def run_all_tests(
        self,
        configs: Optional[List[TestConfig]] = None,
        verbose: bool = True
    ) -> Tuple[int, int, float]:
        """
        Run all test configurations.

        Args:
            configs: List of test configurations. If None, uses default configs.
            verbose: Whether to print detailed results

        Returns:
            Tuple of (passed_count, total_count, average_speedup)
        """
        if configs is None:
            configs = self.create_test_configs()

        self.results = []
        total_tests = len(configs)

        print_header(f"Running {total_tests} Test Cases")

        for i, config in enumerate(configs, 1):
            if verbose:
                print(f"\n[Test {i}/{total_tests}] ", end="")

            result = self.run_single_test(config)
            self.results.append(result)

            if verbose:
                self.print_result(result)

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        avg_speedup = sum(r.speedup for r in self.results if r.passed) / max(passed, 1)

        self.print_summary(passed, total, avg_speedup)

        return passed, total, avg_speedup

    def print_result(self, result: TestResult):
        """Print a single test result."""
        c = result.config

        if result.error_message:
            print_error(f"FAILED")
            print_info(f"Config: bs={c.batch_size}, seq={c.seq_len}, "
                      f"embed={c.embed_dim}, heads={c.num_heads}, bias={c.bias}")
            print_error(f"Error: {result.error_message}")
        elif result.passed:
            print_success(f"PASSED")
            print_info(f"Config: bs={c.batch_size}, seq={c.seq_len}, "
                      f"embed={c.embed_dim}, heads={c.num_heads}, bias={c.bias}")
            print_info(f"Max diff: {result.max_diff:.2e} (tolerance: 1e-4)")
            self.print_performance(result)
        else:
            print_warning(f"FAILED (tolerance exceeded)")
            print_info(f"Config: bs={c.batch_size}, seq={c.seq_len}, "
                      f"embed={c.embed_dim}, heads={c.num_heads}, bias={c.bias}")
            print_warning(f"Max diff: {result.max_diff:.2e} (tolerance: 1e-4)")
            self.print_performance(result)

    def print_performance(self, result: TestResult):
        """Print performance comparison."""
        pytorch_ms = result.pytorch_time_ms
        cuda_ms = result.cuda_time_ms
        speedup = result.speedup

        print_info(f"Performance:")
        print(f"    PyTorch:  {pytorch_ms:.3f} ms")
        print(f"    CUDA:     {cuda_ms:.3f} ms")
        if speedup >= 1:
            print_success(f"    Speedup:  {speedup:.2f}x")
        else:
            print_warning(f"    Speedup:  {speedup:.2f}x (slower)")

    def print_summary(self, passed: int, total: int, avg_speedup: float):
        """Print test summary."""
        print_header("Test Summary")

        print(f"  Total tests:  {total}")
        print(f"  Passed:       {passed}")
        print(f"  Failed:       {total - passed}")
        print(f"  Pass rate:    {100 * passed / total:.1f}%")
        print(f"  Avg speedup:  {avg_speedup:.2f}x")

        if passed == total:
            print_success("\n  All tests passed!")
        elif passed > total * 0.9:
            print_warning("\n  Most tests passed. Check failures above.")
        else:
            print_error("\n  Many tests failed. Review errors above.")

        # Performance summary
        print_header("Performance Analysis")

        speedups = [r.speedup for r in self.results if r.passed]
        if speedups:
            print(f"  Min speedup:  {min(speedups):.2f}x")
            print(f"  Max speedup:  {max(speedups):.2f}x")
            print(f"  Avg speedup:  {sum(speedups) / len(speedups):.2f}x")

            # Find best configuration
            best_result = max(self.results, key=lambda r: r.speedup)
            c = best_result.config
            print_info(f"\n  Best config:")
            print(f"    bs={c.batch_size}, seq={c.seq_len}, "
                  f"embed={c.embed_dim}, heads={c.num_heads}")
            print(f"    Speedup: {best_result.speedup:.2f}x")

    def generate_report(self, output_path: Optional[str] = None):
        """
        Generate a detailed test report.

        Args:
            output_path: Path to save report. If None, prints to stdout.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("Fused Attention Test Report")
        lines.append("=" * 80)
        lines.append("")

        # Summary statistics
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        speedups = [r.speedup for r in self.results if r.passed]

        lines.append("Summary:")
        lines.append(f"  Total tests:   {total}")
        lines.append(f"  Passed:        {passed}")
        lines.append(f"  Failed:        {total - passed}")
        lines.append(f"  Pass rate:     {100 * passed / total:.1f}%")
        if speedups:
            lines.append(f"  Min speedup:   {min(speedups):.2f}x")
            lines.append(f"  Max speedup:   {max(speedups):.2f}x")
            lines.append(f"  Avg speedup:   {sum(speedups) / len(speedups):.2f}x")
        lines.append("")

        # Detailed results
        lines.append("-" * 80)
        lines.append("Detailed Results:")
        lines.append("-" * 80)
        lines.append("")

        headers = ["#", "Batch", "Seq", "Embed", "Heads", "Bias", "Max Diff", "Status",
                   "PyTorch (ms)", "CUDA (ms)", "Speedup"]
        lines.append(f"{' '.join(f'{h:>12}' for h in headers)}")
        lines.append("-" * 130)

        for i, r in enumerate(self.results, 1):
            c = r.config
            status = "PASS" if r.passed else "FAIL"
            bias_str = "Y" if c.bias else "N"

            row = [
                i,
                c.batch_size,
                c.seq_len,
                c.embed_dim,
                c.num_heads,
                bias_str,
                f"{r.max_diff:.2e}",
                status,
                f"{r.pytorch_time_ms:.3f}",
                f"{r.cuda_time_ms:.3f}",
                f"{r.speedup:.2f}x"
            ]
            lines.append(f"{' '.join(f'{str(v):>12}' for v in row)}")

        lines.append("")

        # Failed tests details
        failed = [r for r in self.results if not r.passed]
        if failed:
            lines.append("-" * 80)
            lines.append("Failed Tests:")
            lines.append("-" * 80)
            for r in failed:
                c = r.config
                lines.append(f"Config: bs={c.batch_size}, seq={c.seq_len}, "
                           f"embed={c.embed_dim}, heads={c.num_heads}, bias={c.bias}")
                if r.error_message:
                    lines.append(f"  Error: {r.error_message}")
                else:
                    lines.append(f"  Max diff: {r.max_diff:.2e} (tolerance: 1e-4)")
                lines.append("")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print_info(f"\nReport saved to: {output_path}")
        else:
            print("\n" + report)


def run_specific_test(
    batch_size: int = 2,
    seq_len: int = 128,
    embed_dim: int = 128,
    num_heads: int = 4,
    bias: bool = True
):
    """
    Run a specific test configuration.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        bias: Whether to use bias
    """
    print_header(f"Specific Test: bs={batch_size}, seq={seq_len}, "
                 f"embed={embed_dim}, heads={num_heads}, bias={bias}")

    device = torch.device('cuda') if torch.cuda.is_available() else None
    if device is None:
        print_error("CUDA is not available. Skipping test.")
        return

    suite = AttentionTestSuite(device=device)
    config = TestConfig(batch_size, seq_len, embed_dim, num_heads, bias)
    result = suite.run_single_test(config)
    suite.print_result(result)


def run_quick_test():
    """Run a quick subset of tests for rapid validation."""
    print_header("Quick Test Suite")

    device = torch.device('cuda') if torch.cuda.is_available() else None
    if device is None:
        print_error("CUDA is not available. Skipping test.")
        return

    suite = AttentionTestSuite(device=device)

    # Quick test configurations
    configs = [
        TestConfig(2, 128, 128, 4, True),
        TestConfig(4, 256, 256, 8, True),
        TestConfig(1, 512, 128, 2, False),
    ]

    suite.run_all_tests(configs)


def run_full_test_suite():
    """Run the complete test suite."""
    device = torch.device('cuda') if torch.cuda.is_available() else None
    if device is None:
        print_error("CUDA is not available. Skipping test.")
        return

    suite = AttentionTestSuite(device=device)
    suite.run_all_tests()

    # Generate report
    report_path = Path(__file__).parent.parent / "test_results" / "attention_test_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    suite.generate_report(str(report_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test suite for fused attention kernel"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "full", "specific"],
        default="quick",
        help="Test mode: quick (few tests), full (all tests), or specific"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for specific test")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for specific test")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension for specific test")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of heads for specific test")
    parser.add_argument("--no-bias", action="store_true", help="Disable bias for specific test")

    args = parser.parse_args()

    if args.mode == "quick":
        run_quick_test()
    elif args.mode == "full":
        run_full_test_suite()
    else:
        run_specific_test(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            bias=not args.no_bias
        )
