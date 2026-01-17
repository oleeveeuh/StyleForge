"""
Direct test to verify kernel is being used and QKV projection is correct.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("Direct Kernel Test")
print("=" * 80)

if not torch.cuda.is_available():
    print("CUDA not available!")
    sys.exit(1)

# Configuration
batch_size = 1
seq_len = 2
embed_dim = 32
num_heads = 2
head_dim = 16

# Set seed
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Create test data - use CPU first to verify
x_cpu = torch.randn(batch_size, seq_len, embed_dim)
w_qkv_cpu = torch.randn(3 * embed_dim, embed_dim)
bias_qkv_cpu = torch.randn(3 * embed_dim)

print(f"\nCPU Input x[0,0,:5]: {x_cpu[0,0,:5].tolist()}")
print(f"CPU w_qkv[0,:5]: {w_qkv_cpu[0,:5].tolist()}")

# PyTorch reference on CPU
import torch.nn.functional as F

# Q for head 0: rows 0-15 of w_qkv
w_q_head_0 = w_qkv_cpu[0:16, :]
bias_q_head_0 = bias_qkv_cpu[0:16]
q_pt = F.linear(x_cpu, w_q_head_0, bias_q_head_0)

print(f"\nPyTorch CPU Q[0,0,:5]: {q_pt[0,0,:5].tolist()}")

# Now copy to CUDA and test kernel
x = x_cpu.cuda()
w_qkv = w_qkv_cpu.cuda()
bias_qkv = bias_qkv_cpu.cuda()

print(f"\nCUDA Input x[0,0,:5]: {x[0,0,:5].tolist()}")

# Test with CUDA kernel
from kernels.attention_wrapper import get_attention_module

module = get_attention_module()

# Test the fused_qkv_proj function
try:
    print("\nTesting fused_qkv_proj...")
    qkv_cuda = module.fused_qkv_proj(x, w_qkv, bias_qkv)

    # The output is [batch, seq_len, 3*embed_dim]
    print(f"QKV output shape: {qkv_cuda.shape}")

    # Split into Q, K, V
    q_cuda = qkv_cuda[:, :, 0:embed_dim]
    k_cuda = qkv_cuda[:, :, embed_dim:2*embed_dim]
    v_cuda = qkv_cuda[:, :, 2*embed_dim:3*embed_dim]

    print(f"\nCUDA Q[0,0,:5]: {q_cuda[0,0,:5].tolist()}")
    print(f"CUDA K[0,0,:5]: {k_cuda[0,0,:5].tolist()}")
    print(f"CUDA V[0,0,:5]: {v_cuda[0,0,:5].tolist()}")

    # Compare Q with PyTorch (on CUDA now)
    q_pt_cuda = F.linear(x, w_qkv[:, :16].T, bias_qkv[:16])
    print(f"\nPyTorch CUDA Q[0,0,:5]: {q_pt_cuda[0,0,:5].tolist()}")

    # Actually, PyTorch's F.linear with in_proj_weight is different
    # Let me use the actual MultiheadAttention to verify
    import torch.nn as nn
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=True).cuda()
    with torch.no_grad():
        mha.in_proj_weight.copy_(w_qkv)
        mha.in_proj_bias.copy_(bias_qkv)

    # Get Q, K, V from PyTorch MHA internals
    with torch.no_grad():
        # PyTorch computes QKV internally
        # We can access them through the forward pass with need_weights=False
        qkv_pt = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q_pt_mha = qkv_pt[:, :, 0:embed_dim]
        k_pt_mha = qkv_pt[:, :, embed_dim:2*embed_dim]
        v_pt_mha = qkv_pt[:, :, 2*embed_dim:3*embed_dim]

    print(f"\nPyTorch MHA Q[0,0,:5]: {q_pt_mha[0,0,:5].tolist()}")
    print(f"PyTorch MHA K[0,0,:5]: {k_pt_mha[0,0,:5].tolist()}")
    print(f"PyTorch MHA V[0,0,:5]: {v_pt_mha[0,0,:5].tolist()}")

    # Compare
    q_diff = (q_cuda - q_pt_mha).abs()
    k_diff = (k_cuda - k_pt_mha).abs()
    v_diff = (v_cuda - v_pt_mha).abs()

    print(f"\nQ diff max: {q_diff.max().item():.6e}")
    print(f"K diff max: {k_diff.max().item():.6e}")
    print(f"V diff max: {v_diff.max().item():.6e}")

    if q_diff.max().item() < 1e-4 and k_diff.max().item() < 1e-4 and v_diff.max().item() < 1e-4:
        print("\nPASS: QKV projections match PyTorch MHA!")
    else:
        print("\nFAIL: QKV projections differ from PyTorch MHA")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
