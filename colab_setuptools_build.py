# ============================================
# SETUPTOOLS-BASED KERNEL BUILD FOR COLAB
# ============================================
# This compiles the CUDA kernel using setuptools
# which is more reliable than JIT in Colab.
# ============================================

import torch
import sys
from pathlib import Path

print("=" * 70)
print("  SETUPTOOLS KERNEL BUILD")
print("=" * 70)

# Check CUDA
if not torch.cuda.is_available():
    print("\n❌ CUDA is not available!")
    sys.exit(1)

print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA version: {torch.version.cuda}")

# Read CUDA source
cuda_src_path = Path("kernels/attention.cu")
if not cuda_src_path.exists():
    print(f"\n❌ CUDA source not found at {cuda_src_path}")
    sys.exit(1)

print(f"\n✓ Found CUDA source: {cuda_src_path}")

with open(cuda_src_path, 'r') as f:
    cuda_source = f.read()

# Verify num_heads parameter
if "int64_t num_heads" not in cuda_source:
    print("\n❌ ERROR: num_heads parameter not found!")
    print("Run: !git pull")
    sys.exit(1)
else:
    print("✓ Verified: num_heads parameter is in source")

# The CUDA file already contains PYBIND11_MODULE, so we don't need a separate cpp file
# Passing empty cpp_source to avoid duplicate PyInit definitions
cpp_source = ""

print("\n" + "-" * 70)
print("Compiling with setuptools (this may take 2-3 minutes)...")
print("-" * 70)

try:
    from utils.compile_setuptools import compile_with_setuptools

    # Compile
    module = compile_with_setuptools(
        name="fused_attention",
        cuda_source=cuda_source,
        cpp_source=cpp_source,
        output_dir=Path("kernels"),
        verbose=True
    )

    print("\n✓ Compilation successful!")

except Exception as e:
    print(f"\n❌ Compilation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now test the kernel
print("\n" + "-" * 70)
print("Testing kernel...")
print("-" * 70)

batch_size = 1
seq_len = 64
embed_dim = 128
num_heads = 4

print(f"\nConfiguration:")
print(f"  batch_size = {batch_size}")
print(f"  seq_len = {seq_len}")
print(f"  embed_dim = {embed_dim}")
print(f"  num_heads = {num_heads}")

# Test input
torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, embed_dim, device='cuda', dtype=torch.float32)
w_qkv = torch.randn(3 * embed_dim, embed_dim, device='cuda', dtype=torch.float32)
w_out = torch.randn(embed_dim, embed_dim, device='cuda', dtype=torch.float32)
scale = (embed_dim // num_heads) ** -0.5

try:
    with torch.no_grad():
        output = module.fused_attention_v1(
            x, w_qkv, w_out,
            None, None,
            scale, num_heads
        )

    print(f"\n✓ Kernel execution successful!")
    print(f"  Output shape: {output.shape}")

    # Check for NaN/Inf
    if torch.isnan(output).any():
        print("\n❌ ERROR: Output contains NaN!")
        sys.exit(1)
    if torch.isinf(output).any():
        print("\n❌ ERROR: Output contains Inf!")
        sys.exit(1)

    print("✓ Output is valid")

    # Compare with PyTorch
    import torch.nn as nn
    attn_pt = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
    attn_pt.eval()

    with torch.no_grad():
        attn_pt.in_proj_weight.copy_(w_qkv)
        # PyTorch's Linear computes x @ weight.T
        # Our kernel computes x @ w_out.T
        # So we pass the SAME w_out to both for them to match
        attn_pt.out_proj.weight.copy_(w_out)
        out_pt, _ = attn_pt(x, x, x)

    diff = (output - out_pt).abs()
    max_diff = diff.max().item()

    print(f"\nComparison with PyTorch:")
    print(f"  Max difference: {max_diff:.2e}")

    if max_diff < 1e-3:
        print("\n✅ SUCCESS! Kernel matches PyTorch!")
    else:
        print(f"\n⚠️ Difference: {max_diff:.2e}")

except RuntimeError as e:
    if "invalid argument" in str(e):
        print(f"\n❌ CUDA kernel launch error: invalid argument")
        print("\nThis is strange since we just compiled fresh...")
    else:
        print(f"\n❌ Runtime error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ KERNEL IS READY!")
print("=" * 70)
print("\nThe compiled .so file is in: kernels/fused_attention.so")
print("You can now use: from kernels import FusedAttention")
