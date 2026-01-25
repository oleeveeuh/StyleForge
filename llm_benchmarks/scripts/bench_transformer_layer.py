"""
Benchmark complete transformer layer (Attention + FFN)

This shows end-to-end speedup when both optimized kernels are used together.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from configs.llama2_7b import LLAMA2_7B
from models.custom_attention import CustomMultiHeadAttention, create_pytorch_baseline_attention
from models.custom_ffn import CustomFFN, PyTorchFFN
from models.utils import print_gpu_info
from scripts.benchmark_harness import BenchmarkHarness


class TransformerLayer(nn.Module):
    """
    Single transformer layer: LayerNorm -> Attention -> LayerNorm -> FFN

    Args:
        use_custom_kernels: If True, use optimized CUDA kernels
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        use_custom_kernels: bool = False,
    ):
        super().__init__()

        self.use_custom_kernels = use_custom_kernels

        # Pre-attention LayerNorm
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Attention
        if use_custom_kernels:
            self.attention = CustomMultiHeadAttention(hidden_size, num_heads)
        else:
            self.attention = create_pytorch_baseline_attention(hidden_size, num_heads)

        # Pre-FFN LayerNorm
        self.ffn_norm = nn.LayerNorm(hidden_size)

        # FFN
        if use_custom_kernels:
            self.ffn = CustomFFN(hidden_size, intermediate_size)
        else:
            self.ffn = PyTorchFFN(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections"""

        # Attention block
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        if self.use_custom_kernels:
            attn_output = self.attention(hidden_states)
        else:
            attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)

        hidden_states = residual + attn_output

        # FFN block
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output

        return hidden_states


def benchmark_transformer_layer():
    """
    Benchmark complete transformer layer showing combined speedup
    """
    print("="*70)
    print("Complete Transformer Layer Benchmark - Llama-2-7B")
    print("="*70)

    print_gpu_info()

    config = LLAMA2_7B
    seq_lengths = [512, 1024, 2048, 4096]

    print(f"\nConfiguration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")

    harness = BenchmarkHarness(warmup_iterations=10, benchmark_iterations=50)

    for seq_len in seq_lengths:
        print(f"\n{'='*70}")
        print(f"Sequence Length: {seq_len} tokens")
        print(f"{'='*70}")

        # Create models
        pytorch_layer = TransformerLayer(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            use_custom_kernels=False,
        ).cuda().eval()

        custom_layer = TransformerLayer(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            use_custom_kernels=True,
        ).cuda().eval()

        # Create input
        hidden_states = torch.randn(
            1, seq_len, config.hidden_size,
            dtype=torch.float32, device='cuda'
        )

        # Benchmark
        results = harness.compare(
            baseline_name=f"PyTorch Layer (seq={seq_len})",
            baseline_fn=lambda: pytorch_layer(hidden_states),
            optimized_name=f"Custom Layer (seq={seq_len})",
            optimized_fn=lambda: custom_layer(hidden_states),
        )

        print(f"\n[Analysis] Component breakdown (estimated):")
        print(f"  Attention: ~60% of layer time")
        print(f"  FFN: ~30% of layer time")
        print(f"  LayerNorm: ~10% of layer time")
        print(f"\n  Combined speedup from both kernels: {results['speedup']:.2f}x")

    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    harness.save_results('results/transformer_layer_benchmark.json')
    harness.print_summary()


if __name__ == "__main__":
    benchmark_transformer_layer()
