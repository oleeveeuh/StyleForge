"""
Test that LLM infrastructure is set up correctly

Run this to verify everything is installed and working before benchmarking.
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from configs.llama2_7b import LLAMA2_7B, get_config
from models.utils import (
    create_dummy_attention_inputs,
    create_dummy_ffn_inputs,
    estimate_memory_usage,
    print_gpu_info,
)


def test_setup():
    """Test that all infrastructure is working"""

    print("="*70)
    print("LLM Infrastructure Setup Test")
    print("="*70)

    # Test 1: CUDA availability
    print("\n[Test 1] Checking CUDA...")
    if torch.cuda.is_available():
        print(" CUDA is available")
        print_gpu_info()
    else:
        print(" CUDA not available - benchmarks will run on CPU")
        return

    # Test 2: Configuration loading
    print("\n[Test 2] Loading Llama-2 config...")
    config = LLAMA2_7B
    print(" Config loaded:")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Num heads: {config.num_attention_heads}")
    print(f"   Head dim: {config.head_dim}")
    print(f"   Intermediate size: {config.intermediate_size}")

    # Test 3: Create dummy inputs
    print("\n[Test 3] Creating dummy inputs...")
    seq_len = 512

    Q, K, V = create_dummy_attention_inputs(
        batch_size=1,
        seq_len=seq_len,
        num_heads=config.num_attention_heads,
        head_dim=config.head_dim,
    )
    print(" Created attention inputs:")
    print(f"   Q shape: {Q.shape}")
    print(f"   Device: {Q.device}")

    x, w1, w2 = create_dummy_ffn_inputs(
        batch_size=1,
        seq_len=seq_len,
        hidden_dim=config.hidden_size,
        intermediate_dim=config.intermediate_size,
    )
    print(" Created FFN inputs:")
    print(f"   Input shape: {x.shape}")
    print(f"   W1 shape: {w1.shape}")
    print(f"   W2 shape: {w2.shape}")

    # Test 4: Memory estimation
    print("\n[Test 4] Estimating memory usage...")
    mem = estimate_memory_usage(
        seq_len=seq_len,
        hidden_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        intermediate_dim=config.intermediate_size,
    )
    print(f" Memory estimates for seq_len={seq_len}:")
    print(f"   Attention: {mem['attention_mb']:.2f} MB")
    print(f"   Attention scores: {mem['attention_scores_mb']:.2f} MB")
    print(f"   FFN: {mem['ffn_mb']:.2f} MB")
    print(f"   Total: {mem['total_mb']:.2f} MB")

    # Test 5: Basic PyTorch attention
    print("\n[Test 5] Testing PyTorch attention...")
    try:
        # Standard scaled dot-product attention
        scale = 1.0 / (config.head_dim ** 0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        print(" PyTorch attention works:")
        print(f"   Output shape: {output.shape}")
        print(f"   Output mean: {output.mean().item():.4f}")
        print(f"   Output std: {output.std().item():.4f}")
    except Exception as e:
        print(f" PyTorch attention failed: {e}")
        return

    print("\n" + "="*70)
    print(" All tests passed! Infrastructure is ready for benchmarking.")
    print("="*70)


if __name__ == "__main__":
    test_setup()
