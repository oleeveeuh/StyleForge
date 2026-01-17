"""
Simple test script to run the CUDA kernel with debug prints.

This script runs the kernel with a small configuration and prints
the debug output to compare against expected values.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("CUDA Kernel Debug Test")
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

# Check CUDA availability
if not torch.cuda.is_available():
    print("\nError: CUDA is not available. This test requires a GPU.")
    sys.exit(1)

print(f"\nCUDA device: {torch.cuda.get_device_name(0)}")

# Create input and weights with same seed as Python simulation
torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')
w_qkv = torch.randn(3 * embed_dim, embed_dim, device='cuda')
bias_qkv = torch.randn(3 * embed_dim, device='cuda')
w_out = torch.randn(embed_dim, embed_dim, device='cuda')
bias_out = torch.randn(embed_dim, device='cuda')

# DEBUG: Print what we're about to send to the kernel
print(f"\nDEBUG: Python-side values before calling kernel:")
print(f"  x.shape: {x.shape}, x.stride(): {x.stride()}")
print(f"  x.is_contiguous: {x.is_contiguous()}")
print(f"  x.data_ptr(): {x.data_ptr()}")
print(f"  x[0, 0, 0:5]: {x[0, 0, :5].tolist()}")
print(f"  w_qkv.shape: {w_qkv.shape}, w_qkv.stride(): {w_qkv.stride()}")
print(f"  w_qkv.is_contiguous: {w_qkv.is_contiguous()}")
print(f"  w_qkv[0, 0:5]: {w_qkv[0, :5].tolist()}")
print(f"  bias_qkv[0:5]: {bias_qkv[:5].tolist()}")

# Also print the first 10 values directly from memory
x_cpu = x.cpu()
print(f"  x[0:10] flat: {x_cpu.flatten()[:10].tolist()}")

scale = 1.0 / (head_dim ** 0.5)

print(f"\n" + "=" * 80)
print("Running CUDA kernel with debug prints...")
print("=" * 80)

# Import the FusedAttention module which wraps the CUDA kernel
try:
    from kernels.attention_wrapper import FusedAttention, get_attention_module
except ImportError as e:
    print(f"Error importing FusedAttention: {e}")
    print("\nPlease make sure the CUDA extension is compiled:")
    print("  python setup.py install")
    print("  or: python -m build")
    sys.exit(1)

# Get the low-level module for direct access
try:
    _module = get_attention_module()
    fused_attention_v1 = _module.fused_attention_v1
    print("\nUsing low-level fused_attention_v1 function")
except:
    print("\nFalling back to FusedAttention module...")
    # Use the wrapper module instead
    fused_attention_v1 = None

# Run the kernel
with torch.no_grad():
    if fused_attention_v1 is not None:
        # Direct call to low-level function
        # IMPORTANT: Make sure tensors are contiguous before passing to CUDA
        output_cuda = fused_attention_v1(
            x.contiguous(),
            w_qkv.contiguous(),
            w_out.contiguous(),
            bias_qkv,
            bias_out,
            scale, num_heads
        )
    else:
        # Use the FusedAttention module
        attn = FusedAttention(embed_dim, num_heads).cuda()
        # Copy weights to the module
        with torch.no_grad():
            attn.w_qkv.copy_(w_qkv)
            attn.w_out.copy_(w_out)
            if attn.bias_qkv is not None:
                attn.bias_qkv.copy_(bias_qkv)
            if attn.bias_out is not None:
                attn.bias_out.copy_(bias_out)

        output_cuda = attn(x)

print("\n" + "=" * 80)
print("CUDA kernel completed")
print("=" * 80)

print(f"\nOutput shape: {output_cuda.shape}")
print(f"Output[0, 0, :8]: {output_cuda[0, 0, :8].tolist()}")

print("\n" + "=" * 80)
print("Expected Python values (from print_expected_values.py):")
print("=" * 80)
print("""
Run: python3 kernels/print_expected_values.py
Compare the "=== KERNEL DEBUG ===" output above with the "=== EXPECTED (Python) ===" values.
""")

# Also compute PyTorch reference for comparison
import torch.nn as nn
import torch.nn.functional as F

mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=True).cuda()

# Copy weights to match
with torch.no_grad():
    # Copy in_proj_weight
    mha.in_proj_weight.copy_(w_qkv)
    mha.in_proj_bias.copy_(bias_qkv)
    # Copy out_proj weights
    # Our kernel computes: output = concat_heads @ w_out^T
    # PyTorch's F.linear computes: output = input @ weight^T
    # So to match, we set: PyTorch's weight = our w_out (NOT w_out.T)
    # Then PyTorch computes: concat @ w_out^T which matches our kernel
    mha.out_proj.weight.copy_(w_out)  # FIXED: was w_out.T
    mha.out_proj.bias.copy_(bias_out)

with torch.no_grad():
    output_pt, _ = mha(x, x, x)

print(f"\nPyTorch MHA output[0, 0, :8]: {output_pt[0, 0, :8].tolist()}")

diff = (output_cuda - output_pt).abs()
print(f"\nMax difference: {diff.max().item():.6e}")
print(f"Mean difference: {diff.mean().item():.6e}")

if diff.max().item() < 1e-4:
    print("\n✓ PASS: Output matches PyTorch within tolerance")
else:
    print("\n✗ FAIL: Output differs from PyTorch")
    print("\nCheck the debug output above to find where the computation diverges.")
