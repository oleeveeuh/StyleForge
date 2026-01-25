"""
CPU-based attention benchmark for testing infrastructure

This version doesn't require CUDA and can be used to verify the benchmark
harness works correctly before running on GPU.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from configs.llama2_7b import LLAMA2_7B
from models.utils import validate_attention_output


class CPUAttentionBaseline(nn.Module):
    """CPU-based attention for testing"""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(output)

        return output


def benchmark_cpu_attention():
    """Benchmark CPU attention to verify infrastructure works"""

    print("="*70)
    print("CPU Attention Benchmark (Infrastructure Test)")
    print("="*70)

    config = LLAMA2_7B
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Head dim: {config.head_dim}")

    # Smaller config for CPU testing
    test_config = LLAMA2_7B
    test_config.hidden_size = 512
    test_config.num_attention_heads = 8

    print(f"\nTest Configuration (scaled down for CPU):")
    print(f"  Hidden size: {test_config.hidden_size}")
    print(f"  Num heads: {test_config.num_attention_heads}")
    print(f"  Head dim: {test_config.head_dim}")

    # Create model
    model = CPUAttentionBaseline(
        hidden_size=test_config.hidden_size,
        num_heads=test_config.num_attention_heads,
    ).eval()

    seq_lengths = [64, 128, 256]
    results = []

    print(f"\n{'Seq Len':<10} {'Mean (ms)':<12} {'Median (ms)':<12} {'Std (ms)':<12}")
    print("-"*50)

    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, test_config.hidden_size)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)

        # Benchmark
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        times_array = torch.tensor(times)
        mean_ms = times_array.mean().item()
        median_ms = times_array.median().item()
        std_ms = times_array.std().item()

        print(f"{seq_len:<10} {mean_ms:<12.2f} {median_ms:<12.2f} {std_ms:<12.2f}")

        results.append({
            'seq_len': seq_len,
            'mean_ms': mean_ms,
            'median_ms': median_ms,
            'std_ms': std_ms,
        })

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/attention_cpu_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nCPU benchmark complete. On GPU, expect 50-100x faster speeds.")

    return results


def test_sdpa_availability():
    """Test if PyTorch SDPA is available"""
    print("\n" + "="*70)
    print("PyTorch SDPA Availability Check")
    print("="*70)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Check if SDPA is available
    if hasattr(F, 'scaled_dot_product_attention'):
        print(" scaled_dot_product_attention: AVAILABLE")

        # Check backend
        try:
            backend = torch.backends.cuda.flash_sdp_enabled()
            print(f" Flash Attention backend: {backend}")
        except:
            print(" Flash Attention backend: N/A")

        # Test SDPA
        Q = torch.randn(1, 8, 64, 64)
        K = torch.randn(1, 8, 64, 64)
        V = torch.randn(1, 8, 64, 64)

        try:
            output = F.scaled_dot_product_attention(Q, K, V)
            print(" SDPA test run: SUCCESS")
        except Exception as e:
            print(f" SDPA test run: FAILED ({e})")
    else:
        print(" scaled_dot_product_attention: NOT AVAILABLE")
        print(" (Need PyTorch 2.0+)")


if __name__ == "__main__":
    test_sdpa_availability()
    benchmark_cpu_attention()
