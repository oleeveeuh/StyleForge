# ============================================
# SIMPLE TEST WITH 1 HEAD
# ============================================
# Test with num_heads=1 to eliminate cross-head
# reduction as a source of bugs.
# ============================================

import torch
import sys
from pathlib import Path

print("=" * 70)
print("  TEST WITH 1 HEAD (NO CROSS-HEAD REDUCTION)")
print("=" * 70)

if not torch.cuda.is_available():
    print("\n❌ CUDA is not available!")
    sys.exit(1)

# Test configuration with 1 head
batch_size = 1
seq_len = 64
embed_dim = 128
num_heads = 1  # Only 1 head!

print(f"\nConfiguration:")
print(f"  batch_size = {batch_size}")
print(f"  seq_len = {seq_len}")
print(f"  embed_dim = {embed_dim}")
print(f"  num_heads = {num_heads} (single head, no cross-head reduction)")

torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, embed_dim, device='cuda', dtype=torch.float32)
w_qkv = torch.randn(3 * embed_dim, embed_dim, device='cuda', dtype=torch.float32)
w_out = torch.randn(embed_dim, embed_dim, device='cuda', dtype=torch.float32)
scale = (embed_dim // num_heads) ** -0.5

print(f"\nScale: {scale}")

# Compile kernel
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

    # PyTorch reference
    import torch.nn as nn
    attn_pt = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
    attn_pt.eval()

    with torch.no_grad():
        attn_pt.in_proj_weight.copy_(w_qkv)
        attn_pt.out_proj.weight.copy_(w_out.T)  # PyTorch expects transposed
        out_pt, _ = attn_pt(x, x, x)

    diff = (output - out_pt).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nDifference statistics:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-3:
        print("\n✅ SUCCESS! With 1 head, the kernel matches PyTorch!")
        print("   This suggests the cross-head reduction in the output projection is the bug.")
    else:
        print(f"\n⚠️  Even with 1 head, difference is {max_diff:.2e}")
        print("   This suggests the bug is in the attention computation itself, not the output projection.")

        # Show first few values
        print(f"\nFirst 5 values:")
        print(f"  Kernel:  {output[0, 0, :5]}")
        print(f"  PyTorch: {out_pt[0, 0, :5]}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
