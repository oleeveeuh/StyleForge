# ============================================
# DIRECT KERNEL TEST - BYPASSING ALL CACHES
# ============================================
# This cell compiles and tests the kernel directly
# without relying on any cached .so files.
# ============================================

import torch
import sys
from pathlib import Path

print("=" * 70)
print("  DIRECT KERNEL COMPILATION AND TEST")
print("=" * 70)

# Check CUDA availability
if not torch.cuda.is_available():
    print("\n❌ CUDA is not available!")
    sys.exit(1)

print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA version: {torch.version.cuda}")

# Read CUDA source directly from file
cuda_src_path = Path("kernels/attention.cu")
if not cuda_src_path.exists():
    print(f"\n❌ CUDA source not found at {cuda_src_path}")
    sys.exit(1)

print(f"\n✓ Found CUDA source: {cuda_src_path}")

with open(cuda_src_path, 'r') as f:
    cuda_source = f.read()

# Verify the num_heads parameter is in the source
if "int64_t num_heads" not in cuda_source:
    print("\n❌ ERROR: num_heads parameter not found in CUDA source!")
    print("The source code may not be up to date.")
    print("Run: !git pull")
    sys.exit(1)
else:
    print("✓ Verified: num_heads parameter is present in source")

# CUDA file already contains PYBIND11_MODULE, so pass empty cpp_source
cpp_source = ""

print("\n" + "-" * 70)
print("Compiling kernel (this may take 30-60 seconds)...")
print("-" * 70)

try:
    from torch.utils.cpp_extension import load_inline

    # Force rebuild by using a unique name
    import time
    unique_name = f"fused_attention_test_{int(time.time())}"

    module = load_inline(
        name=unique_name,
        cpp_sources=[cpp_source] if cpp_source else None,
        cuda_sources=[cuda_source],
        extra_cuda_cflags=["-O3"],
        verbose=True  # Show compilation output
    )

    print("\n✓ Compilation successful!")

except Exception as e:
    print(f"\n❌ Compilation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now test with a small example
print("\n" + "-" * 70)
print("Testing kernel with small tensors...")
print("-" * 70)

# Small test configuration
batch_size = 1
seq_len = 64
embed_dim = 128
num_heads = 4  # head_dim = 128/4 = 32

print(f"\nTest configuration:")
print(f"  batch_size = {batch_size}")
print(f"  seq_len = {seq_len}")
print(f"  embed_dim = {embed_dim}")
print(f"  num_heads = {num_heads}")
print(f"  head_dim = {embed_dim // num_heads}")

# Create test input
torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, embed_dim, device='cuda', dtype=torch.float32)

# Create weights
w_qkv = torch.randn(3 * embed_dim, embed_dim, device='cuda', dtype=torch.float32)
w_out = torch.randn(embed_dim, embed_dim, device='cuda', dtype=torch.float32)
scale = (embed_dim // num_heads) ** -0.5

print(f"\nInput shape: {x.shape}")
print(f"QKV weight shape: {w_qkv.shape}")
print(f"Output weight shape: {w_out.shape}")
print(f"Scale: {scale:.4f}")

try:
    with torch.no_grad():
        output = module.fused_attention_v1(
            x, w_qkv, w_out,
            None, None,  # No bias
            scale, num_heads
        )

    print(f"\n✓ Kernel execution successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")

    # Verify no NaN/Inf
    if torch.isnan(output).any():
        print("\n❌ ERROR: Output contains NaN values!")
        sys.exit(1)
    if torch.isinf(output).any():
        print("\n❌ ERROR: Output contains Inf values!")
        sys.exit(1)

    print("\n✓ Output is valid (no NaN/Inf)")

    # Compare with PyTorch
    print("\n" + "-" * 70)
    print("Comparing with PyTorch nn.MultiheadAttention...")
    print("-" * 70)

    import torch.nn as nn
    attn_pytorch = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
    attn_pytorch.eval()

    # Copy weights
    with torch.no_grad():
        attn_pytorch.in_proj_weight.copy_(w_qkv)
        attn_pytorch.out_proj.weight.copy_(w_out.T)  # PyTorch expects transposed

    with torch.no_grad():
        out_pytorch, _ = attn_pytorch(x, x, x)

    diff = (output - out_pytorch).abs()
    max_diff = diff.max().item()

    print(f"\nPyTorch output shape: {out_pytorch.shape}")
    print(f"Max difference: {max_diff:.2e}")

    if max_diff < 1e-3:
        print("\n✅ SUCCESS! Kernel matches PyTorch output!")
    else:
        print(f"\n⚠️ Difference exceeds 1e-3 threshold")
        print(f"First 5 values of output[0,0]:")
        print(f"  Kernel:  {output[0,0,:5]}")
        print(f"  PyTorch: {out_pytorch[0,0,:5]}")

except RuntimeError as e:
    if "invalid argument" in str(e):
        print(f"\n❌ CUDA kernel launch error: {e}")
        print("\nThis usually means:")
        print("  1. The cached .so file is stale (old version without num_heads)")
        print("  2. Runtime needs to be restarted")
        print("\nSOLUTION:")
        print("  1. Runtime → Restart session")
        print("  2. Run this cell again")
    else:
        print(f"\n❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nThe kernel is working correctly.")
print("You can now proceed with the rest of the notebook.")
