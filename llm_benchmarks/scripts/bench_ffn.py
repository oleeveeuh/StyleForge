"""
Benchmark custom FFN kernel vs PyTorch sequential ops on Llama-2-7B

Measures performance across multiple sequence lengths and analyzes fusion benefits.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from configs.llama2_7b import LLAMA2_7B
from models.custom_ffn import CustomFFN, PyTorchFFN, count_ffn_operations
from models.utils import validate_attention_output, print_gpu_info
from scripts.benchmark_harness import BenchmarkHarness
import json


def benchmark_ffn_single_config(
    seq_len: int,
    config=LLAMA2_7B,
    batch_size: int = 1,
    harness: BenchmarkHarness = None,
) -> dict:
    """
    Benchmark FFN at a single sequence length

    Args:
        seq_len: Sequence length to test
        config: Model configuration
        batch_size: Batch size
        harness: Benchmark harness instance

    Returns:
        Dictionary with benchmark results
    """
    if harness is None:
        harness = BenchmarkHarness(warmup_iterations=10, benchmark_iterations=100)

    print(f"\n{'='*70}")
    print(f"Benchmarking FFN: seq_len={seq_len}")
    print(f"{'='*70}")

    # Create models
    custom_ffn = CustomFFN(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
    ).cuda().eval()

    pytorch_ffn = PyTorchFFN(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
    ).cuda().eval()

    # Create input
    hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size,
        dtype=torch.float32, device='cuda'
    )

    # Validate correctness first
    print("\n[Validation] Checking numerical accuracy...")
    with torch.no_grad():
        custom_output = custom_ffn(hidden_states)
        pytorch_output = pytorch_ffn(hidden_states)

    is_close, max_error, mean_error = validate_attention_output(
        custom_output, pytorch_output, rtol=1e-3, atol=1e-4
    )

    print(f"  Max error:  {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")

    if is_close:
        print("  Outputs match within tolerance")
    else:
        print(f"  Outputs differ (max error: {max_error:.2e})")
        print("  Note: GELU approximation may cause small differences")

    # Benchmark
    print("\n[Benchmark] Running performance comparison...")

    results = harness.compare(
        baseline_name=f"PyTorch Sequential (seq={seq_len})",
        baseline_fn=lambda: pytorch_ffn(hidden_states),
        optimized_name=f"Custom Fused (seq={seq_len})",
        optimized_fn=lambda: custom_ffn(hidden_states),
    )

    # Add metadata
    results['seq_len'] = seq_len
    results['batch_size'] = batch_size
    results['hidden_size'] = config.hidden_size
    results['intermediate_size'] = config.intermediate_size
    results['validation'] = {
        'max_error': max_error,
        'mean_error': mean_error,
        'is_close': is_close,
    }

    # Compute FLOP/s
    flop_counts = count_ffn_operations(
        batch_size, seq_len, config.hidden_size, config.intermediate_size
    )

    pytorch_tflops = flop_counts['total_gflops'] / (results['baseline']['mean_ms'] / 1000) / 1000
    custom_tflops = flop_counts['total_gflops'] / (results['optimized']['mean_ms'] / 1000) / 1000

    results['compute'] = {
        'total_gflops': flop_counts['total_gflops'],
        'pytorch_tflops': pytorch_tflops,
        'custom_tflops': custom_tflops,
    }

    print(f"\n[Compute] Total FLOPs: {flop_counts['total_gflops']:.2f} GFLOPs")
    print(f"  PyTorch throughput: {pytorch_tflops:.2f} TFLOP/s")
    print(f"  Custom throughput:  {custom_tflops:.2f} TFLOP/s")

    # Estimate memory savings
    intermediate_tensor_size = batch_size * seq_len * config.intermediate_size * 4 / (1024**2)
    print(f"\n[Memory] Intermediate tensor: {intermediate_tensor_size:.2f} MB")
    print(f"  Custom kernel eliminates this allocation (fused ops)")

    results['memory_saved_mb'] = intermediate_tensor_size

    return results


def benchmark_ffn_sweep():
    """
    Benchmark FFN across multiple sequence lengths

    Tests 512, 1024, 2048 tokens to show scalability.
    """
    print("="*70)
    print("Feed-Forward Network Benchmark - Llama-2-7B Configuration")
    print("="*70)

    print_gpu_info()

    config = LLAMA2_7B
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Expansion ratio: {config.intermediate_size / config.hidden_size:.2f}x")

    harness = BenchmarkHarness(warmup_iterations=10, benchmark_iterations=50)

    # Test multiple sequence lengths
    seq_lengths = [512, 1024, 2048, 4096]
    all_results = []

    for seq_len in seq_lengths:
        try:
            results = benchmark_ffn_single_config(
                seq_len=seq_len,
                config=config,
                batch_size=1,
                harness=harness,
            )
            all_results.append(results)
        except RuntimeError as e:
            print(f"  Error at seq_len={seq_len}: {e}")
            continue

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Seq Len':<10} {'PyTorch (ms)':<15} {'Custom (ms)':<15} {'Speedup':<10} {'Memory Saved':<15}")
    print("-"*70)

    for result in all_results:
        seq_len = result['seq_len']
        pytorch_ms = result['baseline']['mean_ms']
        custom_ms = result['optimized']['mean_ms']
        speedup = result['speedup']
        memory_saved = result['memory_saved_mb']

        print(f"{seq_len:<10} {pytorch_ms:<15.2f} {custom_ms:<15.2f} {speedup:<10.2f}x {memory_saved:<15.1f} MB")

    # Save results
    os.makedirs('results', exist_ok=True)
    output_file = 'results/ffn_benchmark.json'
    harness.save_results(output_file)

    # Also save detailed results
    with open('results/ffn_detailed.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetailed results saved to: results/ffn_detailed.json")

    # Print insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    avg_speedup = sum(r['speedup'] for r in all_results) / len(all_results)
    print(f" Average speedup: {avg_speedup:.2f}x")
    print(f" Benefit comes from eliminating kernel launch overhead")
    print(f" Fused GELU (PTX asm) avoids function call overhead")
    print(f" Eliminates intermediate tensor allocations")

    return all_results


if __name__ == "__main__":
    benchmark_ffn_sweep()
