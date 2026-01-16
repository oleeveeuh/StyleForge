# ============================================
# MINIMAL TEST - CHECK QKV PROJECTION ONLY
# ============================================
# This test isolates the QKV projection to check
# if it matches PyTorch's computation.
# ============================================

import torch
import sys
from pathlib import Path

print("=" * 70)
print("  MINIMAL QKV PROJECTION TEST")
print("=" * 70)

if not torch.cuda.is_available():
    print("\n❌ CUDA is not available!")
    sys.exit(1)

print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")

# Very small test
batch_size = 1
seq_len = 2
embed_dim = 16
num_heads = 2
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

# First, let's manually verify the QKV projection is correct
print("\n" + "-" * 70)
print("Step 1: Verify QKV projection manually")
print("-" * 70)

# For head 0, position 0:
x_00 = x[0, 0, :]  # [embed_dim]

# Q weights for head 0
w_q_head0 = w_qkv[0:head_dim, :]  # [head_dim, embed_dim]

# Compute Q manually
q_manual = x_00 @ w_q_head0.T
print(f"\nManual Q[head=0, pos=0]: {q_manual}")

# Now use the kernel
try:
    from torch.utils.cpp_extension import load_inline
    import time

    # Read CUDA source
    cuda_src_path = Path("kernels/attention.cu")
    with open(cuda_src_path, 'r') as f:
        cuda_source = f.read()

    # C++ wrapper
    cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_attention_v1(
    torch::Tensor x,
    torch::Tensor w_qkv,
    torch::Tensor w_out,
    torch::optional<torch::Tensor> bias_qkv,
    torch::optional<torch::Tensor> bias_out,
    float scale,
    int64_t num_heads
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_attention_v1", &fused_attention_v1, "Fused Multi-Head Attention V1");
}
"""

    print("\nCompiling kernel...")
    unique_name = f"test_fused_attention_{int(time.time())}"

    module = load_inline(
        name=unique_name,
        cpp_sources=[cpp_source],
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
    print(f"Kernel output[0, 0]: {output[0, 0]}")

    # PyTorch reference
    import torch.nn as nn
    attn_pt = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
    attn_pt.eval()

    with torch.no_grad():
        attn_pt.in_proj_weight.copy_(w_qkv)
        attn_pt.out_proj.weight.copy_(w_out)
        out_pt, _ = attn_pt(x, x, x)

    print(f"\nPyTorch output[0, 0]: {out_pt[0, 0]}")

    diff = (output - out_pt).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nDifference statistics:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-3:
        print("\n✅ SUCCESS! Kernel matches PyTorch!")
    else:
        print(f"\n⚠️ Difference exceeds threshold")
        print(f"\nFirst 5 values comparison:")
        print(f"  Kernel:  {output[0, 0, :5]}")
        print(f"  PyTorch: {out_pt[0, 0, :5]}")

        # Check if values are scaled differently
        kernel_norm = output[0, 0].abs().max().item()
        pytorch_norm = out_pt[0, 0].abs().max().item()
        print(f"\nMax absolute values:")
        print(f"  Kernel:  {kernel_norm:.4f}")
        print(f"  PyTorch: {pytorch_norm:.4f}")
        if kernel_norm > 0 and pytorch_norm > 0:
            print(f"  Ratio: {kernel_norm / pytorch_norm:.4f}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
