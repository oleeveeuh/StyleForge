#!/usr/bin/env python3
"""
Colab-compatible debug script for CUDA kernel.

This script can be run in a Jupyter notebook or Colab to debug
the CUDA kernel with printf output.

Usage in Colab:
1. Mount drive or upload files
2. Run this cell
3. Compare printf output with expected values
"""

import torch
import sys
import os

# Add StyleForge to path
if 'StyleForge' in os.listdir('/content/drive/MyDrive'):
    sys.path.insert(0, '/content/drive/MyDrive/StyleForge')
else:
    sys.path.insert(0, '.')

print("=" * 80)
print("CUDA Kernel Debug - Colab Version")
print("=" * 80)

# Configuration - MUST match print_expected_values.py
torch.manual_seed(42)
batch_size = 1
seq_len = 4
embed_dim = 32
num_heads = 2
head_dim = embed_dim // num_heads

print(f"\nConfiguration:")
print(f"  batch_size = {batch_size}")
print(f"  seq_len = {seq_len}")
print(f"  embed_dim = {embed_dim}")
print(f"  num_heads = {num_heads}")
print(f"  head_dim = {head_dim}")

# Check CUDA
if not torch.cuda.is_available():
    print("\n⚠️ Warning: CUDA not available. Using CPU for reference only.")

# Create input and weights
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = torch.randn(batch_size, seq_len, embed_dim, device=device)
w_qkv = torch.randn(3 * embed_dim, embed_dim, device=device)
bias_qkv = torch.randn(3 * embed_dim, device=device)
w_out = torch.randn(embed_dim, embed_dim, device=device)
bias_out = torch.randn(embed_dim, device=device)

scale = 1.0 / (head_dim ** 0.5)

print(f"\n" + "=" * 80)
print("Expected Values (Python simulation):")
print("=" * 80)

# Compute expected values using PyTorch operations
import torch.nn.functional as F

# Q for head 0, pos 0
w_q_head_0 = w_qkv[0:head_dim, :]
bias_q_head_0 = bias_qkv[0:head_dim]
q_head_0_pos_0 = F.linear(x[0:1, 0:1, :], w_q_head_0, bias_q_head_0)[0, 0, :]

print(f"x[0,0,0:5]: {x[0,0,0]:.6f} {x[0,0,1]:.6f} {x[0,0,2]:.6f} {x[0,0,3]:.6f} {x[0,0,4]:.6f}")
print(f"w_q_offset: 0")
print(f"w_q[0,0:5]: {w_q_head_0[0,0]:.6f} {w_q_head_0[0,1]:.6f} {w_q_head_0[0,2]:.6f} {w_q_head_0[0,3]:.6f} {w_q_head_0[0,4]:.6f}")
print(f"bias_q[0:5]: {bias_q_head_0[0]:.6f} {bias_q_head_0[1]:.6f} {bias_q_head_0[2]:.6f} {bias_q_head_0[3]:.6f} {bias_q_head_0[4]:.6f}")
print(f"q_reg[0:5]: {q_head_0_pos_0[0]:.6f} {q_head_0_pos_0[1]:.6f} {q_head_0_pos_0[2]:.6f} {q_head_0_pos_0[3]:.6f} {q_head_0_pos_0[4]:.6f}")

# K for head 0, pos 0
w_k_head_0 = w_qkv[embed_dim:embed_dim+head_dim, :]
bias_k_head_0 = bias_qkv[embed_dim:embed_dim+head_dim]
k_head_0_pos_0 = F.linear(x[0:1, 0:1, :], w_k_head_0, bias_k_head_0)[0, 0, :]

print(f"w_k_offset: {embed_dim * embed_dim}")
print(f"w_k[0,0:5]: {w_k_head_0[0,0]:.6f} {w_k_head_0[0,1]:.6f} {w_k_head_0[0,2]:.6f} {w_k_head_0[0,3]:.6f} {w_k_head_0[0,4]:.6f}")
print(f"k_reg[0:5]: {k_head_0_pos_0[0]:.6f} {k_head_0_pos_0[1]:.6f} {k_head_0_pos_0[2]:.6f} {k_head_0_pos_0[3]:.6f} {k_head_0_pos_0[4]:.6f}")

# V for head 0, pos 0
w_v_head_0 = w_qkv[2*embed_dim:2*embed_dim+head_dim, :]
bias_v_head_0 = bias_qkv[2*embed_dim:2*embed_dim+head_dim]
v_head_0_pos_0 = F.linear(x[0:1, 0:1, :], w_v_head_0, bias_v_head_0)[0, 0, :]

print(f"w_v_offset: {2 * embed_dim * embed_dim}")
print(f"w_v[0,0:5]: {w_v_head_0[0,0]:.6f} {w_v_head_0[0,1]:.6f} {w_v_head_0[0,2]:.6f} {w_v_head_0[0,3]:.6f} {w_v_head_0[0,4]:.6f}")
print(f"v_reg[0:5]: {v_head_0_pos_0[0]:.6f} {v_head_0_pos_0[1]:.6f} {v_head_0_pos_0[2]:.6f} {v_head_0_pos_0[3]:.6f} {v_head_0_pos_0[4]:.6f}")

# Attention score
raw_score = (q_head_0_pos_0 @ k_head_0_pos_0).item()
scaled_score = raw_score * scale
print(f"raw_score (Q.K^T): {raw_score:.6f}")
print(f"scaled_score: {scaled_score:.6f}")

if torch.cuda.is_available():
    print(f"\n" + "=" * 80)
    print("Running CUDA kernel (look for '=== KERNEL DEBUG ===' output)...")
    print("=" * 80)

    # Import and run kernel
    try:
        from styleforge import fused_attention_v1

        with torch.no_grad():
            output_cuda = fused_attention_v1(
                x, w_qkv, w_out,
                bias_qkv, bias_out,
                scale, num_heads
            )

        print(f"\n" + "=" * 80)
        print("CUDA kernel completed")
        print("=" * 80)
        print(f"\nOutput shape: {output_cuda.shape}")
        print(f"Output[0, 0, :8]: {output_cuda[0, 0, :8].tolist()}")

    except Exception as e:
        print(f"Error running CUDA kernel: {e}")
        import traceback
        traceback.print_exc()

    # PyTorch reference
    print(f"\n" + "=" * 80)
    print("PyTorch nn.MultiheadAttention reference:")
    print("=" * 80)

    import torch.nn as nn
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=True).cuda()

    with torch.no_grad():
        mha.in_proj_weight.copy_(w_qkv)
        mha.in_proj_bias.copy_(bias_qkv)
        mha.out_proj.weight.copy_(w_out.T)
        mha.out_proj.bias.copy_(bias_out)

        output_pt, _ = mha(x, x, x)

    print(f"PyTorch output[0, 0, :8]: {output_pt[0, 0, :8].tolist()}")

    diff = (output_cuda - output_pt).abs()
    print(f"\nMax difference: {diff.max().item():.6e}")
    print(f"Mean difference: {diff.mean().item():.6e}")

    if diff.max().item() < 1e-4:
        print("\n✓ PASS")
    else:
        print("\n✗ FAIL")
else:
    print("\n⚠️ CUDA not available - cannot run kernel test")

print("\n" + "=" * 80)
print("Debug Complete")
print("=" * 80)
