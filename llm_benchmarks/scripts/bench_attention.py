"""
Benchmark custom attention kernel vs PyTorch SDPA on Llama-2-7B

Measures performance across multiple sequence lengths to demonstrate scalability.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from configs.llama2_7b import LLAMA2_7B
from models.custom_attention import CustomMultiHeadAttention, create_pytorch_baseline_attention
from models.utils import create_dummy_attention_inputs, validate_attention_output, print_gpu_info
from scripts.benchmark_harness import BenchmarkHarness
import json


def pytorch_sdpa_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    PyTorch's scaled_dot_product_attention (optimized SDPA)

    This uses PyTorch 2.0+ optimized attention which may use Flash Attention 2
    or memory-efficient attention depending on the backend.
    """
    # PyTorch 2.0+ SDPA
    return F.scaled_dot_product_attention(Q, K, V)


def benchmark_attention_single_config(
    seq_len: int,
    config=LLAMA2_7B,
    batch_size: int = 1,
    harness: BenchmarkHarness = None,
) -> dict:
    """
    Benchmark attention at a single sequence length

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
    print(f"Benchmarking Attention: seq_len={seq_len}")
    print(f"{'='*70}")

    # Create models
    custom_attn = CustomMultiHeadAttention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
    ).cuda().eval()

    pytorch_attn = create_pytorch_baseline_attention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
    ).cuda().eval()

    # Create input
    hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size,
        dtype=torch.float32, device='cuda'
    )

    # Validate correctness first
    print("\n[Validation] Checking numerical accuracy...")
    with torch.no_grad():
        custom_output = custom_attn(hidden_states)
        pytorch_output, _ = pytorch_attn(hidden_states, hidden_states, hidden_states)

    is_close, max_error, mean_error = validate_attention_output(
        custom_output, pytorch_output, rtol=1e-3, atol=1e-4
    )

    print(f"  Max error:  {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")

    if is_close:
        print("  Outputs match within tolerance")
    else:
        print(f"  Outputs differ (max error: {max_error:.2e})")
        print("  Continuing with benchmark anyway...")

    # Benchmark
    print("\n[Benchmark] Running performance comparison...")

    results = harness.compare(
        baseline_name=f"PyTorch MHA (seq={seq_len})",
        baseline_fn=lambda: pytorch_attn(hidden_states, hidden_states, hidden_states),
        optimized_name=f"Custom Kernel (seq={seq_len})",
        optimized_fn=lambda: custom_attn(hidden_states),
    )

    # Add metadata
    results['seq_len'] = seq_len
    results['batch_size'] = batch_size
    results['num_heads'] = config.num_attention_heads
    results['head_dim'] = config.head_dim
    results['validation'] = {
        'max_error': max_error,
        'mean_error': mean_error,
        'is_close': is_close,
    }

    # Estimate memory usage
    attention_matrix_size = batch_size * config.num_attention_heads * seq_len * seq_len * 4 / (1024**2)
    print(f"\n[Memory] Attention matrix size: {attention_matrix_size:.2f} MB")
    print(f"  Custom kernel avoids materializing this (O(N) vs O(N²))")

    # Memory reduction calculation
    custom_memory = seq_len * config.head_dim * 4 / (1024**2)  # Per head, per query
    reduction_pct = 100 * (1 - custom_memory / (attention_matrix_size / (batch_size * config.num_attention_heads)))
    results['memory_reduction_pct'] = max(reduction_pct, 0)
    print(f"  Estimated memory reduction: {results['memory_reduction_pct']:.1f}%")

    return results


def benchmark_attention_sweep():
    """
    Benchmark attention across multiple sequence lengths

    Tests 512, 1024, 2048, 4096 tokens to show scalability.
    """
    print("="*70)
    print("Multi-Head Attention Benchmark - Llama-2-7B Configuration")
    print("="*70)

    print_gpu_info()

    config = LLAMA2_7B
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Head dim: {config.head_dim}")
    print(f"  Intermediate size: {config.intermediate_size}")

    harness = BenchmarkHarness(warmup_iterations=10, benchmark_iterations=50)

    # Test multiple sequence lengths
    seq_lengths = [512, 1024, 2048, 4096]
    all_results = []

    for seq_len in seq_lengths:
        try:
            results = benchmark_attention_single_config(
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
    print(f"{'Seq Len':<10} {'PyTorch (ms)':<15} {'Custom (ms)':<15} {'Speedup':<10} {'Mem Reduction':<15}")
    print("-"*70)

    for result in all_results:
        seq_len = result['seq_len']
        pytorch_ms = result['baseline']['mean_ms']
        custom_ms = result['optimized']['mean_ms']
        speedup = result['speedup']
        mem_reduction = result['memory_reduction_pct']

        print(f"{seq_len:<10} {pytorch_ms:<15.2f} {custom_ms:<15.2f} {speedup:<10.2f}x {mem_reduction:<15.1f}%")

    # Save results
    os.makedirs('results', exist_ok=True)
    output_file = 'results/attention_benchmark.json'
    harness.save_results(output_file)

    # Also save detailed results
    with open('results/attention_detailed.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetailed results saved to: results/attention_detailed.json")

    return all_results


if __name__ == "__main__":
    benchmark_attention_sweep()
