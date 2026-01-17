"""
Test to verify QKV projection fix independently.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("QKV Projection Test")
print("=" * 80)

if not torch.cuda.is_available():
    print("CUDA not available!")
    sys.exit(1)

# Configuration
batch_size = 1
seq_len = 4
embed_dim = 32
num_heads = 2
head_dim = 16

# Set seed
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Create test data
x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')
w_qkv = torch.randn(3 * embed_dim, embed_dim, device='cuda')
bias_qkv = torch.randn(3 * embed_dim, device='cuda')

print(f"\nx[0,0,:5]: {x[0,0,:5].tolist()}")
print(f"w_qkv[0,:5]: {w_qkv[0,:5].tolist()}")

# Test with PyTorch
import torch.nn.functional as F

# Q for head 0: rows 0-15 of w_qkv
w_q_head_0 = w_qkv[0:16, :]
bias_q_head_0 = bias_qkv[0:16]
q_pt = F.linear(x, w_q_head_0, bias_q_head_0)

print(f"\nPyTorch Q[0,0,:5]: {q_pt[0,0,:5].tolist()}")

# K for head 0: rows 32-47 of w_qkv
w_k_head_0 = w_qkv[32:48, :]
bias_k_head_0 = bias_qkv[32:48]
k_pt = F.linear(x, w_k_head_0, bias_k_head_0)

print(f"PyTorch K[0,0,:5]: {k_pt[0,0,:5].tolist()}")

# V for head 0: rows 64-79 of w_qkv
w_v_head_0 = w_qkv[64:80, :]
bias_v_head_0 = bias_qkv[64:80]
v_pt = F.linear(x, w_v_head_0, bias_v_head_0)

print(f"PyTorch V[0,0,:5]: {v_pt[0,0,:5].tolist()}")

# Test with CUDA kernel
from kernels.attention_wrapper import get_attention_module

module = get_attention_module()

# The kernel should have a fused_qkv_proj function
# Let's test it directly
try:
    qkv_cuda = module.fused_qkv_proj(x, w_qkv, bias_qkv)

    # The output is [batch, seq_len, 3*embed_dim]
    # Split into Q, K, V
    q_cuda = qkv_cuda[:, :, 0:embed_dim]
    k_cuda = qkv_cuda[:, :, embed_dim:2*embed_dim]
    v_cuda = qkv_cuda[:, :, 2*embed_dim:3*embed_dim]

    print(f"\nCUDA Q[0,0,:5]: {q_cuda[0,0,:5].tolist()}")
    print(f"CUDA K[0,0,:5]: {k_cuda[0,0,:5].tolist()}")
    print(f"CUDA V[0,0,:5]: {v_cuda[0,0,:5].tolist()}")

    # Compare
    q_diff = (q_cuda - q_pt).abs()
    k_diff = (k_cuda - k_pt).abs()
    v_diff = (v_cuda - v_pt).abs()

    print(f"\nMax difference Q: {q_diff.max().item():.6e}")
    print(f"Max difference K: {k_diff.max().item():.6e}")
    print(f"Max difference V: {v_diff.max().item():.6e}")

    if q_diff.max().item() < 1e-4 and k_diff.max().item() < 1e-4 and v_diff.max().item() < 1e-4:
        print("\nPASS: QKV projections match PyTorch!")
    else:
        print("\nFAIL: QKV projections differ from PyTorch")

except Exception as e:
    print(f"\nError testing fused_qkv_proj: {e}")
    print("The function might not be available or there's an issue with the kernel")
