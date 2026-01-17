# ============================================
# INTERMEDIATE VALUE DEBUG TEST
# ============================================
# This test compares intermediate outputs between
# the kernel and PyTorch to find where they diverge.
# ============================================

import torch
import sys
from pathlib import Path

print("=" * 70)
print("  INTERMEDIATE VALUE DEBUG TEST")
print("=" * 70)

if not torch.cuda.is_available():
    print("\n❌ CUDA is not available!")
    sys.exit(1)

print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")

# Small test configuration for detailed debugging
batch_size = 1
seq_len = 4  # Very small for easier debugging
embed_dim = 32  # Small
num_heads = 2  # Small
head_dim = embed_dim // num_heads

print(f"\nConfiguration:")
print(f"  batch_size = {batch_size}")
print(f"  seq_len = {seq_len}")
print(f"  embed_dim = {embed_dim}")
print(f"  num_heads = {num_heads}")
print(f"  head_dim = {head_dim}")

torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, embed_dim, device='cuda', dtype=torch.float32)
w_qkv = torch.randn(3 * embed_dim, embed_dim, device='cuda', dtype=torch.float32)
w_out = torch.randn(embed_dim, embed_dim, device='cuda', dtype=torch.float32)
scale = (embed_dim // num_heads) ** -0.5

print(f"\nScale: {scale}")

# Compile and run kernel
try:
    from torch.utils.cpp_extension import load_inline
    import time

    cuda_src_path = Path("kernels/attention.cu")
    with open(cuda_src_path, 'r') as f:
        cuda_source = f.read()

    # CUDA file already contains PYBIND11_MODULE, so pass empty list for cpp_sources
    cpp_sources = []

    print("\nCompiling kernel...")
    unique_name = f"test_fused_attention_{int(time.time())}"

    module = load_inline(
        name=unique_name,
        cpp_sources=cpp_sources,
        cuda_sources=[cuda_source],
        extra_cuda_cflags=["-O3"],
        verbose=False
    )

    print("✓ Compilation successful")

    with torch.no_grad():
        output = module.fused_attention_v1(
            x, w_qkv, w_out,
            None, None,
            scale, num_heads
        )

    print(f"\nKernel output shape: {output.shape}")

    # PyTorch reference with detailed intermediate values
    import torch.nn as nn
    attn_pt = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
    attn_pt.eval()

    with torch.no_grad():
        attn_pt.in_proj_weight.copy_(w_qkv)
        attn_pt.out_proj.weight.copy_(w_out.T)  # PyTorch expects transposed

        # Get Q, K, V projections
        # PyTorch's in_proj_weight is [3*embed_dim, embed_dim]
        # Q = x @ w_q.T, K = x @ w_k.T, V = x @ w_v.T
        # where w_q, w_k, w_v are each [embed_dim, embed_dim]

        # Split the weight
        w_q_pt = attn_pt.in_proj_weight[:embed_dim, :]  # [embed_dim, embed_dim]
        w_k_pt = attn_pt.in_proj_weight[embed_dim:2*embed_dim, :]
        w_v_pt = attn_pt.in_proj_weight[2*embed_dim:, :]

        # For multi-head, each weight matrix is split across heads
        # Head h uses rows [h*head_dim:(h+1)*head_dim]

        # Compute Q, K, V for all positions
        # Shape: [batch, seq_len, embed_dim]
        Q_all = x @ w_q_pt.T  # [batch, seq, embed_dim]
        K_all = x @ w_k_pt.T
        V_all = x @ w_v_pt.T

        print(f"\nPyTorch Q shape: {Q_all.shape}")
        print(f"PyTorch Q[0, 0]: {Q_all[0, 0]}")

        # Reshape for multi-head: [batch, seq, num_heads, head_dim]
        Q_heads = Q_all.view(batch_size, seq_len, num_heads, head_dim)
        K_heads = K_all.view(batch_size, seq_len, num_heads, head_dim)
        V_heads = V_all.view(batch_size, seq_len, num_heads, head_dim)

        # Compute attention for each head
        head_outputs = []

        for h in range(num_heads):
            # Get Q, K, V for this head
            Q_h = Q_heads[:, :, h, :]  # [batch, seq, head_dim]
            K_h = K_heads[:, :, h, :]
            V_h = V_heads[:, :, h, :]

            # Compute attention scores: Q @ K.T / scale
            attn_scores = (Q_h @ K_h.transpose(-2, -1)) * scale  # [batch, seq, seq]
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # Apply to V
            head_out = attn_weights @ V_h  # [batch, seq, head_dim]
            head_outputs.append(head_out)

            print(f"\nHead {h} output[0, 0]: {head_out[0, 0]}")

        # Concatenate heads
        concat = torch.cat(head_outputs, dim=-1)  # [batch, seq, embed_dim]
        print(f"\nConcatenated[0, 0]: {concat[0, 0]}")

        # Output projection
        out_pt_manual = concat @ attn_pt.out_proj.weight.T
        print(f"Manual output[0, 0]: {out_pt_manual[0, 0]}")

        # Using PyTorch's forward
        out_pt, _ = attn_pt(x, x, x)
        print(f"PyTorch output[0, 0]: {out_pt[0, 0]}")

    diff = (output - out_pt).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nDifference statistics:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    # Check if kernel output matches intermediate values
    print(f"\nKernel output[0, 0]: {output[0, 0]}")
    print(f"PyTorch output[0, 0]: {out_pt[0, 0]}")

    # Check if kernel output is scaled differently
    kernel_max = output.abs().max().item()
    pytorch_max = out_pt.abs().max().item()

    print(f"\nMax absolute values:")
    print(f"  Kernel:  {kernel_max:.6f}")
    print(f"  PyTorch: {pytorch_max:.6f}")

    if kernel_max > 0 and pytorch_max > 0:
        ratio = kernel_max / pytorch_max
        print(f"  Ratio (kernel/pytorch): {ratio:.6f}")

        # Check if ratio is close to num_heads
        if abs(ratio - num_heads) < 0.1:
            print(f"\n⚠️  WARNING: Ratio is close to num_heads ({num_heads})")
            print("     This suggests only ONE head's output is being used!")

    # Check per-head comparison
    print(f"\nDetailed per-element comparison (first 8 elements):")
    for i in range(8):
        k_val = output[0, 0, i].item()
        p_val = out_pt[0, 0, i].item()
        diff_val = abs(k_val - p_val)
        print(f"  [{i}] Kernel: {k_val:8.4f}, PyTorch: {p_val:8.4f}, Diff: {diff_val:8.4f}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
