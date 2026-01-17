"""
Diagnostic script to analyze the OUTPUT PROJECTION weight layout.

The output projection in MultiheadAttention:
1. Concatenates all head outputs: [head_0, head_1, ..., head_N] -> [batch, seq, embed_dim]
2. Applies output projection: output = concat @ w_out.T + bias_out

This script verifies our kernel's output projection matches PyTorch's.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("Output Projection Weight Layout Diagnostic")
print("=" * 80)

# Configuration
embed_dim = 32
num_heads = 2
head_dim = embed_dim // num_heads  # 16
batch_size = 1
seq_len = 4

print(f"\nConfiguration:")
print(f"  embed_dim = {embed_dim}")
print(f"  num_heads = {num_heads}")
print(f"  head_dim = {head_dim}")

# Create a multi-head attention layer
torch.manual_seed(42)
mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=True)

# Create input
torch.manual_seed(123)
x = torch.randn(batch_size, seq_len, embed_dim)

# Get PyTorch's output
with torch.no_grad():
    attn_output_pt, _ = mha(x, x, x)

print(f"\nPyTorch MHA output shape: {attn_output_pt.shape}")
print(f"PyTorch MHA output[0, 0, :8]: {attn_output_pt[0, 0, :8].tolist()}")

# Extract weights
in_proj_weight = mha.in_proj_weight
in_proj_bias = mha.in_proj_bias
# PyTorch's MHA stores output projection in out_proj (a Linear layer)
out_proj_weight = mha.out_proj.weight  # [embed_dim, embed_dim]
out_proj_bias = mha.out_proj.bias if mha.out_proj.bias is not None else None

print(f"\n" + "=" * 80)
print("Step 1: Understand PyTorch's output projection")
print("=" * 80)

print(f"\nout_proj_weight shape: {out_proj_weight.shape}")
print(f"out_proj_bias shape: {out_proj_bias.shape if out_proj_bias is not None else 'None'}")

# PyTorch's nn.MultiheadAttention stores out_proj_weight as:
# - If bias=True: out_proj_weight is [embed_dim, embed_dim]
# - If bias=False (and not using fused projection): might be different

print(f"\nout_proj_weight is {type(out_proj_weight)}")

# Check if it's a Parameter or a matrix with separate bias
if isinstance(out_proj_weight, nn.Parameter):
    print(f"  It's a Parameter with shape {out_proj_weight.shape}")
    # For nn.Linear, the weight is [out_features, in_features]
    # So out_proj @ concat gives output
    # This means: output = concat @ out_proj_weight.T + bias

print("\n" + "=" * 80)
print("Step 2: Manually compute QKV projection and split into heads")
print("=" * 80)

# Full QKV projection
qkv_full = F.linear(x, in_proj_weight, in_proj_bias)  # [1, 4, 96]
q_full, k_full, v_full = qkv_full.chunk(3, dim=-1)

# Split into heads
q_heads = q_full.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
k_heads = k_full.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
v_heads = v_full.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

print(f"Q_heads shape: {q_heads.shape}")  # [1, 2, 4, 16]
print(f"K_heads shape: {k_heads.shape}")
print(f"V_heads shape: {v_heads.shape}")

print("\n" + "=" * 80)
print("Step 3: Compute attention manually (per-head)")
print("=" * 80)

scale = 1.0 / (head_dim ** 0.5)
print(f"Scale: {scale}")

# Compute attention for each head
head_outputs = []

for h in range(num_heads):
    # Q, K, V for this head
    q_h = q_heads[:, h, :, :]  # [1, 4, 16]
    k_h = k_heads[:, h, :, :]  # [1, 4, 16]
    v_h = v_heads[:, h, :, :]  # [1, 4, 16]

    # Attention scores: Q @ K.T
    attn_scores = (q_h @ k_h.transpose(-2, -1)) * scale  # [1, 4, 4]
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Weighted sum of V
    head_out = attn_weights @ v_h  # [1, 4, 16]
    head_outputs.append(head_out)

print(f"\nHead outputs:")
for h, ho in enumerate(head_outputs):
    print(f"  Head {h} shape: {ho.shape}, pos 0: {ho[0, 0, :4].tolist()}")

print("\n" + "=" * 80)
print("Step 4: Concatenate heads")
print("=" * 80)

# Concatenate along the last dimension
# head_outputs is a list of [1, 4, 16] tensors
# Concatenating gives [1, 4, 32]
concat_heads = torch.cat(head_outputs, dim=-1)

print(f"Concatenated heads shape: {concat_heads.shape}")
print(f"concat[0, 0, :8]: {concat_heads[0, 0, :8].tolist()}")

print("\n" + "=" * 80)
print("Step 5: Apply output projection")
print("=" * 80)

print(f"\nout_proj_weight:\n{out_proj_weight}")

# The key question: How does PyTorch apply the output projection?
# For nn.Linear, it's: output = input @ weight.T + bias
# So: output = concat_heads @ out_proj_weight.T + out_proj_bias

output_manual = F.linear(concat_heads, out_proj_weight, out_proj_bias)

print(f"\nManual output shape: {output_manual.shape}")
print(f"Manual output[0, 0, :8]: {output_manual[0, 0, :8].tolist()}")

print(f"\nDoes it match PyTorch's MHA output?")
print(f"  Match: {torch.allclose(output_manual, attn_output_pt, atol=1e-5)}")

if not torch.allclose(output_manual, attn_output_pt, atol=1e-5):
    print(f"\n  Max difference: {(output_manual - attn_output_pt).abs().max().item()}")

print("\n" + "=" * 80)
print("Step 6: Verify kernel's output projection approach")
print("=" * 80)

print("""
The kernel's output_projection_kernel computes:
  output[out_dim] = sum_over_heads(head_output[head] @ w_out[out_dim, head*head_dim:(head+1)*head_dim])

This is equivalent to:
  output = concat_heads @ w_out.T

Let's verify this manually:
""")

# Manual computation following kernel's logic
output_kernel_like = torch.zeros(batch_size, seq_len, embed_dim)

for out_dim in range(embed_dim):
    for head_idx in range(num_heads):
        # Extract head output
        head_out = head_outputs[head_idx]  # [1, 4, 16]

        # Extract relevant portion of w_out
        # w_out[out_dim, head_idx*head_dim:(head_idx+1)*head_dim]
        w_out_slice = out_proj_weight[out_dim, head_idx*head_dim:(head_idx+1)*head_dim]

        # Compute dot product
        partial = (head_out * w_out_slice).sum(dim=-1)  # [1, 4]

        output_kernel_like[:, :, out_dim] += partial

# Add bias
if out_proj_bias is not None:
    output_kernel_like += out_proj_bias

print(f"Kernel-like output[0, 0, :8]: {output_kernel_like[0, 0, :8].tolist()}")

print(f"\nDoes kernel-like output match manual output?")
print(f"  Match: {torch.allclose(output_kernel_like, output_manual, atol=1e-5)}")

print(f"\nDoes kernel-like output match PyTorch's MHA output?")
print(f"  Match: {torch.allclose(output_kernel_like, attn_output_pt, atol=1e-5)}")

print("\n" + "=" * 80)
print("Step 7: Trace kernel's indexing")
print("=" * 80)

print("""
Kernel code (output_projection_kernel):

    // head_outputs layout: [batch, num_heads, seq_len, head_dim]
    int64_t head_offset = ((int64_t)batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM + seq_idx * HEAD_DIM;
    const float* head_ptr = head_outputs + head_offset;

    // w_out layout: [embed_dim, embed_dim], row out_dim has w_out[out_dim, :]
    int64_t w_out_offset = (int64_t)out_dim * embed_dim + head_idx * HEAD_DIM;
    const float* w_out_ptr = w_out + w_out_offset;

For batch_idx=0, head_idx=0, seq_idx=0, out_dim=0, HEAD_DIM=16, embed_dim=32:
  head_offset = (0 * 2 + 0) * 4 * 16 + 0 * 16 = 0
  w_out_offset = 0 * 32 + 0 * 16 = 0

For batch_idx=0, head_idx=1, seq_idx=0, out_dim=0:
  head_offset = (0 * 2 + 1) * 4 * 16 + 0 * 16 = 64
  w_out_offset = 0 * 32 + 1 * 16 = 16

Let's verify the w_out indexing:
""")

# Verify w_out indexing
print(f"\nw_out shape: {out_proj_weight.shape}")
print(f"For out_dim=0, head_idx=0: w_out_offset=0, slice=w_out[0, 0:16]")
print(f"  w_out[0, 0:16] = {out_proj_weight[0, :head_dim].tolist()}")
print(f"\nFor out_dim=0, head_idx=1: w_out_offset=16, slice=w_out[0, 16:32]")
print(f"  w_out[0, 16:32] = {out_proj_weight[0, head_dim:].tolist()}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if torch.allclose(output_kernel_like, attn_output_pt, atol=1e-5):
    print("\n✓ The kernel's output projection approach is CORRECT!")
    print("  The indexing and computation match PyTorch's implementation.")
else:
    print("\n✗ MISMATCH in output projection!")
    print(f"  Max difference: {(output_kernel_like - attn_output_pt).abs().max().item()}")
    print("\nPossible issues:")
    print("  1. w_out matrix layout is different than expected")
    print("  2. Head outputs buffer layout is different")
    print("  3. Reduction across heads is not working correctly")

print("\n" + "=" * 80)
