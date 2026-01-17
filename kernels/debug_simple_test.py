"""
Minimal direct test of the CUDA kernel with guaranteed seed.
"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force everything fresh
torch.cuda.empty_cache()

# Set seed BEFORE importing anything else
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("=" * 80)
print("MINIMAL CUDA KERNEL TEST")
print("=" * 80)

# Check CUDA
if not torch.cuda.is_available():
    print("CUDA not available!")
    sys.exit(1)

print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Configuration
batch_size = 1
seq_len = 4
embed_dim = 32
num_heads = 2
head_dim = 16

# Create tensors with FRESH seed
# Set seed again right before creation
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')
w_qkv = torch.randn(3 * embed_dim, embed_dim, device='cuda')
bias_qkv = torch.randn(3 * embed_dim, device='cuda')
w_out = torch.randn(embed_dim, embed_dim, device='cuda')
bias_out = torch.randn(embed_dim, device='cuda')

print(f"\nValues created with seed 42:")
print(f"  x[0,0,0:5] = {x[0,0,:5].tolist()}")
print(f"  w_qkv[0,0:5] = {w_qkv[0,:5].tolist()}")

# Import and run kernel
print(f"\nImporting kernel...")
from kernels.attention_wrapper import get_attention_module

_module = get_attention_module()
fused_attention_v1 = _module.fused_attention_v1

print(f"Running kernel...")
scale = 1.0 / (head_dim ** 0.5)

with torch.no_grad():
    output_cuda = fused_attention_v1(
        x.contiguous(),
        w_qkv.contiguous(),
        w_out.contiguous(),
        bias_qkv,
        bias_out,
        scale,
        num_heads
    )

print(f"\nOutput shape: {output_cuda.shape}")
print(f"Output[0,0,:5] = {output_cuda[0,0,:5].tolist()}")

# PyTorch reference
print(f"\nComputing PyTorch reference...")
import torch.nn as nn

# Reset seed again for PyTorch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Recreate tensors for PyTorch (same seed)
x_pt = torch.randn(batch_size, seq_len, embed_dim, device='cuda')
w_qkv_pt = torch.randn(3 * embed_dim, embed_dim, device='cuda')
bias_qkv_pt = torch.randn(3 * embed_dim, device='cuda')
w_out_pt = torch.randn(embed_dim, embed_dim, device='cuda')
bias_out_pt = torch.randn(embed_dim, device='cuda')

mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=True).cuda()

with torch.no_grad():
    mha.in_proj_weight.copy_(w_qkv_pt)
    mha.in_proj_bias.copy_(bias_qkv_pt)
    mha.out_proj.weight.copy_(w_out_pt.T)
    mha.out_proj.bias.copy_(bias_out_pt)

    output_pt, _ = mha(x_pt, x_pt, x_pt)

print(f"PyTorch x[0,0,:5] = {x_pt[0,0,:5].tolist()}")
print(f"PyTorch output[0,0,:5] = {output_pt[0,0,:5].tolist()}")

# Compare
diff = (output_cuda - output_pt).abs()
print(f"\nMax diff: {diff.max().item():.6e}")

if diff.max().item() < 1e-4:
    print("PASS!")
else:
    print("FAIL!")
    print(f"\nThis means the kernel computation is wrong (not just a seed issue).")
