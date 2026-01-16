# ============================================
# SIMPLE IMPORT TEST - NO RECOMPILATION
# ============================================
# This test just tries to import and use the
# existing kernels without recompiling.
# ============================================

import torch
import sys
from pathlib import Path

print("=" * 70)
print("  SIMPLE KERNEL IMPORT TEST")
print("=" * 70)

# Check CUDA
if not torch.cuda.is_available():
    print("\n❌ CUDA is not available!")
    sys.exit(1)

print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
print(f"✓ PyTorch version: {torch.__version__}")

# First, check if the source file has num_heads
cuda_src_path = Path("kernels/attention.cu")
if not cuda_src_path.exists():
    print(f"\n❌ CUDA source not found at {cuda_src_path}")
    sys.exit(1)

with open(cuda_src_path, 'r') as f:
    cuda_source = f.read()

if "int64_t num_heads" not in cuda_source:
    print("\n❌ ERROR: num_heads parameter not found in CUDA source!")
    print("Run: !git pull")
    sys.exit(1)
else:
    print("✓ Verified: num_heads parameter is in source code")

# Check the wrapper
wrapper_path = Path("kernels/attention_wrapper.py")
if not wrapper_path.exists():
    print(f"\n❌ Wrapper not found at {wrapper_path}")
    sys.exit(1)

with open(wrapper_path, 'r') as f:
    wrapper_content = f.read()

# Verify wrapper passes num_heads
if "num_heads)" not in wrapper_content or "module.fused_attention_v1" not in wrapper_content:
    print("\n❌ ERROR: Wrapper may not be passing num_heads correctly")
    sys.exit(1)
else:
    print("✓ Verified: Wrapper calls fused_attention_v1 with num_heads")

# Now try to import
print("\n" + "-" * 70)
print("Attempting to import kernels...")
print("-" * 70)

try:
    from kernels import FusedAttention
    print("✓ Import successful!")

except Exception as e:
    print(f"\n❌ Import failed: {e}")
    print("\nThis is expected if the kernel hasn't been compiled yet.")
    print("The kernel needs to be JIT-compiled first.")
    sys.exit(1)

# Now test the kernel
print("\n" + "-" * 70)
print("Testing kernel with small tensors...")
print("-" * 70)

# Small test configuration
batch_size = 1
seq_len = 64
embed_dim = 128
num_heads = 4  # head_dim = 32

print(f"\nTest configuration:")
print(f"  batch_size = {batch_size}")
print(f"  seq_len = {seq_len}")
print(f"  embed_dim = {embed_dim}")
print(f"  num_heads = {num_heads}")

try:
    # Create model
    model = FusedAttention(embed_dim, num_heads).cuda()
    model.eval()

    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')

    # Run
    with torch.no_grad():
        output = model(x)

    print(f"\n✓ Kernel execution successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")

    # Check for NaN/Inf
    if torch.isnan(output).any():
        print("\n❌ ERROR: Output contains NaN!")
        sys.exit(1)
    if torch.isinf(output).any():
        print("\n❌ ERROR: Output contains Inf!")
        sys.exit(1)

    # Compare with PyTorch
    print("\n" + "-" * 70)
    print("Comparing with PyTorch...")
    print("-" * 70)

    import torch.nn as nn
    attn_pt = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
    attn_pt.eval()

    # Copy the weights from our model to PyTorch for fair comparison
    with torch.no_grad():
        # Split fused QKV weight
        w_q = model.w_qkv[:embed_dim]
        w_k = model.w_qkv[embed_dim:2*embed_dim]
        w_v = model.w_qkv[2*embed_dim:3*embed_dim]

        # PyTorch expects concatenated Q, K, V weights
        attn_pt.in_proj_weight.copy_(torch.cat([w_q, w_k, w_v], dim=0))
        # PyTorch out_proj expects transposed weight
        attn_pt.out_proj.weight.copy_(model.w_out.T)

        # Copy biases if present
        if model.bias_qkv is not None and attn_pt.in_proj_bias is not None:
            attn_pt.in_proj_bias.copy_(model.bias_qkv)
        if model.bias_out is not None and attn_pt.out_proj.bias is not None:
            attn_pt.out_proj.bias.copy_(model.bias_out)

    with torch.no_grad():
        out_pt, _ = attn_pt(x, x, x)

    diff = (output - out_pt).abs()
    max_diff = diff.max().item()

    print(f"\nPyTorch output shape: {out_pt.shape}")
    print(f"Max difference: {max_diff:.2e}")

    if max_diff < 1e-3:
        print("\n✅ SUCCESS! Kernel matches PyTorch!")
    else:
        print(f"\n⚠️ Difference: {max_diff:.2e} (expected < 1e-3)")
        print(f"First 5 values:")
        print(f"  Kernel:  {output[0,0,:5]}")
        print(f"  PyTorch: {out_pt[0,0,:5]}")

except RuntimeError as e:
    error_str = str(e)
    if "invalid argument" in error_str:
        print(f"\n❌ CUDA kernel launch error: invalid argument")
        print("\nThis usually means the cached .so file is stale (old version)")
        print("without the num_heads parameter.")
        print("\nSOLUTION:")
        print("  1. Clear the cache")
        print("  2. Restart the runtime")
        print("  3. Run this test again")
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
