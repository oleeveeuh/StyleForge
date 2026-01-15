"""
StyleForge - PyTorch Profiler Utilities

Utilities for profiling CUDA kernels with PyTorch Profiler.
"""

import torch
from torch.profiler import profile, ProfilerActivity, record_function, tensorboard_trace_handler
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import json


class KernelProfiler:
    """
    Profile CUDA kernels using PyTorch Profiler.

    Features:
    - CPU and CUDA timing
    - Memory usage tracking
    - Chrome trace export
    - TensorBoard integration
    """

    def __init__(
        self,
        activities: list = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False
    ):
        """
        Args:
            activities: List of profiler activities (default: CPU and CUDA)
            record_shapes: Record tensor shapes
            profile_memory: Profile memory usage
            with_stack: Record Python stack traces
        """
        self.activities = activities or [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

    def profile(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        name: str = "kernel",
        warmup_iters: int = 5,
        profile_iters: int = 10,
        export_chrome_trace: Optional[Path] = None,
        export_tensorboard: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Profile a model with given input.

        Args:
            model: PyTorch model to profile
            input_data: Input tensor
            name: Name for the profiling session
            warmup_iters: Number of warmup iterations
            profile_iters: Number of profile iterations
            export_chrome_trace: Path to save Chrome trace
            export_tensorboard: Path to save TensorBoard logs

        Returns:
            Dictionary with profiling results
        """
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(input_data)

        torch.cuda.synchronize()

        # Profile
        with profile(
            activities=self.activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack
        ) as prof:
            with record_function(name):
                with torch.no_grad():
                    for _ in range(profile_iters):
                        _ = model(input_data)

        # Extract metrics
        results = self._extract_metrics(prof)

        # Export traces
        if export_chrome_trace:
            export_chrome_trace = Path(export_chrome_trace)
            export_chrome_trace.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(export_chrome_trace))
            results['chrome_trace'] = str(export_chrome_trace)

        if export_tensorboard:
            export_tensorboard = Path(export_tensorboard)
            export_tensorboard.parent.mkdir(parents=True, exist_ok=True)
            prof.export_memory_timeline(str(export_tensorboard))
            results['tensorboard_log'] = str(export_tensorboard)

        return results

    def _extract_metrics(self, prof: profile) -> Dict[str, Any]:
        """Extract metrics from profiler object."""
        events = prof.key_averages()

        cuda_events = [e for e in events if e.cuda_time_total > 0]
        memory_events = [e for e in events if e.cuda_memory_usage > 0]

        total_cuda_time = sum(e.cuda_time_total for e in cuda_events)
        total_cpu_time = sum(e.cpu_time_total for e in events)

        return {
            'total_cuda_time_us': total_cuda_time,
            'total_cpu_time_us': total_cpu_time,
            'cuda_kernel_count': len(cuda_events),
            'top_cuda_kernels': [
                {
                    'name': e.key,
                    'cuda_time_us': e.cuda_time_total,
                    'cuda_time_ms': e.cuda_time_total / 1000,
                    'cpu_time_us': e.cpu_time_total,
                    'cuda_memory_mb': e.cuda_memory_usage / 1e6,
                    'calls': e.count
                }
                for e in sorted(cuda_events, key=lambda x: -x.cuda_time_total)[:10]
            ],
            'memory_usage_mb': sum(e.cuda_memory_usage for e in memory_events) / 1e6
        }

    def print_summary(self, results: Dict[str, Any]):
        """Print formatted profiling summary."""
        print("\n" + "=" * 70)
        print(f"  PROFILING SUMMARY")
        print("=" * 70)

        print(f"\nTotal CUDA Time: {results['total_cuda_time_us'] / 1000:.2f} ms")
        print(f"Total CPU Time:  {results['total_cpu_time_us'] / 1000:.2f} ms")
        print(f"Kernels Executed: {results['cuda_kernel_count']}")

        if results['top_cuda_kernels']:
            print("\n" + "-" * 70)
            print("  TOP CUDA KERNELS")
            print("-" * 70)
            print(f"{'Kernel':<40} {'Time (ms)':<12} {'Calls':<8} {'Memory (MB)':<12}")
            print("-" * 70)

            for kernel in results['top_cuda_kernels']:
                name = kernel['name'][:38] + '..' if len(kernel['name']) > 40 else kernel['name']
                print(f"{name:<40} {kernel['cuda_time_ms']:<12.2f} {kernel['calls']:<8} {kernel['cuda_memory_mb']:<12.2f}")

        if results.get('chrome_trace'):
            print(f"\nChrome Trace: {results['chrome_trace']}")
            print("  View at: chrome://tracing")

        print("=" * 70 + "\n")


def profile_attention_comparison(
    models: Dict[str, torch.nn.Module],
    input_shape: tuple,
    output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Profile multiple attention implementations for comparison.

    Args:
        models: Dictionary of {name: model} to profile
        input_shape: Shape tuple for input tensor (batch, seq, embed)
        output_dir: Directory to save profiling outputs

    Returns:
        Dictionary of profiling results for each model
    """
    profiler = KernelProfiler()

    results = {}
    x = torch.randn(*input_shape, device='cuda')

    for name, model in models.items():
        print(f"\n🔍 Profiling {name}...")
        model = model.cuda().eval()

        result = profiler.profile(
            model=model,
            input_data=x,
            name=name,
            warmup_iters=5,
            profile_iters=20,
            export_chrome_trace=output_dir / f"{name}_trace.json" if output_dir else None
        )

        results[name] = result
        profiler.print_summary(result)

    # Comparison summary
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("  COMPARISON SUMMARY")
        print("=" * 70)

        baseline = results[list(models.keys())[0]]
        baseline_time = baseline['total_cuda_time_us']

        print(f"\n{'Model':<20} {'CUDA Time (ms)':<15} {'Speedup':<10}")
        print("-" * 70)

        for name, result in results.items():
            time_ms = result['total_cuda_time_us'] / 1000
            speedup = baseline_time / result['total_cuda_time_us']
            marker = ' (baseline)' if name == list(models.keys())[0] else ''
            print(f"{name:<20} {time_ms:<15.2f} {speedup:<10.2f}x{marker}")

        print("=" * 70 + "\n")

    return results


def benchmark_with_profiler(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    name: str = "model"
):
    """
    Quick benchmark with profiler output.

    Args:
        model: Model to benchmark
        input_data: Input tensor
        name: Model name for output
    """
    profiler = KernelProfiler()

    results = profiler.profile(
        model=model,
        input_data=input_data,
        name=name,
        warmup_iters=10,
        profile_iters=50
    )

    profiler.print_summary(results)

    return results


def save_profiling_results(results: Dict[str, Any], filepath: Path):
    """
    Save profiling results to JSON file.

    Args:
        results: Results dictionary from profiler
        filepath: Path to save JSON
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {filepath}")


if __name__ == "__main__":
    # Test profiling
    from kernels import FusedAttention, FusedAttentionV2

    models = {
        "PyTorch": torch.nn.MultiheadAttention(128, 4, batch_first=True),
        "V1": FusedAttention(128, 4),
        "V2": FusedAttentionV2(128, 4)
    }

    results = profile_attention_comparison(
        models=models,
        input_shape=(2, 256, 128),
        output_dir=Path('benchmarks')
    )
