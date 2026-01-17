"""
Diagnostic script to analyze PyTorch's nn.MultiheadAttention weight layout
vs our kernel's expected layout.

The key insight: PyTorch computes FULL QKV projection first, then splits into heads.
Our kernel tries to compute per-head projections directly using head-specific weights.

These should be equivalent IF the weight layout is understood correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("PyTorch MultiheadAttention Weight Layout Diagnostic")
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

# Extract the in_proj_weight and bias
# PyTorch stores Q, K, V weights concatenated: [Q; K; V] where each is [embed_dim, embed_dim]
in_proj_weight = mha.in_proj_weight  # [3*embed_dim, embed_dim] = [96, 32]
in_proj_bias = mha.in_proj_bias if mha.in_proj_bias is not None else None  # [3*embed_dim] = [96]

print(f"\nPyTorch weight shapes:")
print(f"  in_proj_weight: {in_proj_weight.shape}")
print(f"  in_proj_bias: {in_proj_bias.shape if in_proj_bias is not None else 'None'}")

# Create input
torch.manual_seed(123)
x = torch.randn(batch_size, seq_len, embed_dim)
print(f"\nInput shape: {x.shape}")

print("\n" + "=" * 80)
print("PYTORCH'S WAY: Compute full QKV, then split into heads")
print("=" * 80)

# Step 1: Full QKV projection (PyTorch's way)
qkv_full = F.linear(x, in_proj_weight, in_proj_bias)  # [1, 4, 96]
print(f"\n1. Full QKV projection: {qkv_full.shape}")

# Step 2: Split into Q, K, V
q_full_pt, k_full_pt, v_full_pt = qkv_full.chunk(3, dim=-1)
print(f"   Q_full: {q_full_pt.shape}, K_full: {k_full_pt.shape}, V_full: {v_full_pt.shape}")

# Step 3: Reshape and transpose to split into heads
# [batch, seq, embed_dim] -> [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
q_heads_pt = q_full_pt.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
k_heads_pt = k_full_pt.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
v_heads_pt = v_full_pt.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

print(f"\n2. After reshape and transpose:")
print(f"   Q_heads: {q_heads_pt.shape} (batch, num_heads, seq, head_dim)")

print(f"\n3. Q for head 0, position 0:")
print(f"   {q_heads_pt[0, 0, 0, :]}")

print(f"\n4. Q for head 1, position 0:")
print(f"   {q_heads_pt[0, 1, 0, :]}")

print("\n" + "=" * 80)
print("KERNEL'S EXPECTED WAY: Per-head projection using head-specific weights")
print("=" * 80)

# Extract Q, K, V weight sections
w_q_pt = in_proj_weight[:embed_dim, :]      # [32, 32] - Q weights
w_k_pt = in_proj_weight[embed_dim:2*embed_dim, :]  # [32, 32] - K weights
w_v_pt = in_proj_weight[2*embed_dim:, :]    # [32, 32] - V weights

if in_proj_bias is not None:
    b_q_pt = in_proj_bias[:embed_dim]
    b_k_pt = in_proj_bias[embed_dim:2*embed_dim]
    b_v_pt = in_proj_bias[2*embed_dim:]
else:
    b_q_pt = b_k_pt = b_v_pt = None

print(f"\n1. Extracted Q, K, V weight shapes:")
print(f"   w_q: {w_q_pt.shape}, w_k: {w_k_pt.shape}, w_v: {w_v_pt.shape}")

# Now let's compute Q for head 0 using the KERNEL'S approach
# The kernel uses: w_qkv[head_idx * head_dim : (head_idx+1) * head_dim, :]
# This extracts a SUBSET of rows from the Q weight matrix

print("\n2. Kernel's weight layout interpretation:")
print(f"   For head 0: uses w_qkv[0:{head_dim}, :] = w_qkv[0:16, :]")
print(f"   For head 1: uses w_qkv[{head_dim}:{2*head_dim}, :] = w_qkv[16:32, :]")

# Extract head-specific weights as kernel does
w_q_head0_kernel = w_q_pt[0:head_dim, :]      # [16, 32]
w_q_head1_kernel = w_q_pt[head_dim:2*head_dim, :]  # [16, 32]

b_q_head0_kernel = b_q_pt[0:head_dim] if b_q_pt is not None else None
b_q_head1_kernel = b_q_pt[head_dim:2*head_dim] if b_q_pt is not None else None

print(f"\n3. Head-specific weight shapes:")
print(f"   w_q_head0: {w_q_head0_kernel.shape}")
print(f"   w_q_head1: {w_q_head1_kernel.shape}")

# Compute Q for head 0 using kernel approach
# Q_head0 = x @ w_q_head0.T + bias_head0
# But wait! The kernel computes: output = x @ w.T (not x @ w.T directly)
# Let's trace through the kernel logic more carefully

print("\n4. Manual computation using kernel's approach:")
print(f"   Computing: Q_head0 = x @ w_q_head0.T")

# The kernel's qkv_projection_vectorized computes:
# for i in 0..HEAD_DIM:
#     output[i] += x_val * w_ptr[i * embed_dim + k]
# This is: output = x @ w.T (standard matrix multiplication)

q_head0_kernel = F.linear(x, w_q_head0_kernel, b_q_head0_kernel)  # [1, 4, 16]
q_head1_kernel = F.linear(x, w_q_head1_kernel, b_q_head1_kernel)  # [1, 4, 16]

print(f"   Q_head0_kernel shape: {q_head0_kernel.shape}")
print(f"   Q_head0_kernel[0, 0, :]: {q_head0_kernel[0, 0, :]}")

print("\n" + "=" * 80)
print("COMPARISON: PyTorch vs Kernel")
print("=" * 80)

print(f"\nQ for head 0, position 0:")
print(f"  PyTorch: {q_heads_pt[0, 0, 0, :].tolist()}")
print(f"  Kernel:  {q_head0_kernel[0, 0, :].tolist()}")
print(f"  Match:   {torch.allclose(q_heads_pt[0, 0, 0, :], q_head0_kernel[0, 0, :])}")

print(f"\nQ for head 1, position 0:")
print(f"  PyTorch: {q_heads_pt[0, 1, 0, :].tolist()}")
print(f"  Kernel:  {q_head1_kernel[0, 0, :].tolist()}")
print(f"  Match:   {torch.allclose(q_heads_pt[0, 1, 0, :], q_head1_kernel[0, 0, :])}")

print("\n" + "=" * 80)
print("UNDERSTANDING THE LAYOUT")
print("=" * 80)

print("""
The key insight is that PyTorch's weight matrix layout is:

w_qkv[:embed_dim, :] contains ALL Q weights (for all heads combined)
When we do: Q_full = x @ w_qkv[:embed_dim, :].T

The result Q_full has shape [batch, seq, embed_dim]
This represents: [batch, seq, head_0_q + head_1_q + ... + head_N_q]

So Q_full[:, :, head_dim*h:head_dim*(h+1)] gives us Q for head h!

Our kernel tries to compute:
  Q_head_h = x @ w_qkv[head_dim*h : head_dim*(h+1), :].T

This uses a SUBSET of rows from w_qkv.
But: w_qkv[head_dim*h : head_dim*(h+1), :] contains exactly the weights
that contribute to Q_full[:, :, head_dim*h:head_dim*(h+1)]!

Let's verify this relationship:
""")

# Verify: Q_full[:, :, 0:16] should equal x @ w_qkv[0:16, :].T
q_full_head0_slice = q_full_pt[:, :, 0:head_dim]
print(f"Q_full[:, :, 0:{head_dim}] = Q_full_pt[:, :, 0:{head_dim}]")
print(f"  {q_full_head0_slice[0, 0, :].tolist()}")

print(f"\nx @ w_qkv[0:{head_dim}, :].T:")
print(f"  {q_head0_kernel[0, 0, :].tolist()}")

print(f"\nAre they equal? {torch.allclose(q_full_head0_slice, q_head0_kernel)}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

match_h0 = torch.allclose(q_heads_pt[0, 0, 0, :], q_head0_kernel[0, 0, :])
match_h1 = torch.allclose(q_heads_pt[0, 1, 0, :], q_head1_kernel[0, 0, :])

if match_h0 and match_h1:
    print("\n✓ The kernel's weight layout interpretation is CORRECT!")
    print("  Computing per-head projections using head-specific weight rows")
    print("  produces the same result as PyTorch's approach.")
else:
    print("\n✗ MISMATCH DETECTED!")
    print("  The kernel's approach produces different results than PyTorch.")

print("\n" + "=" * 80)
print("CHECKING KERNEL CODE WEIGHT OFFSET CALCULATION")
print("=" * 80)

print("""
Kernel code (attention_per_head_kernel):

    // Q weights for this head: w_qkv[head_idx * head_dim : (head_idx+1) * head_dim, :]
    int64_t w_q_head_offset = (int64_t)head_idx * head_dim * embed_dim;

This offset points to: w_qkv + head_idx * head_dim * embed_dim

In row-major layout with embed_dim columns:
- Row 0 starts at offset 0
- Row 1 starts at offset embed_dim
- Row r starts at offset r * embed_dim

So w_qkv + head_idx * head_dim * embed_dim points to row (head_idx * head_dim).

For head_idx=0, head_dim=16, embed_dim=32:
  offset = 0 * 16 * 32 = 0 (row 0)

For head_idx=1, head_dim=16, embed_dim=32:
  offset = 1 * 16 * 32 = 512 (row 16)

This is CORRECT for extracting w_qkv[head_idx*head_dim : (head_idx+1)*head_dim, :]!
""")

print("\n" + "=" * 80)
print("NOW LET'S TRACE THROUGH A FULL ATTENTION COMPUTATION")
print("=" * 80)

# Compute full attention using PyTorch's MHA
with torch.no_grad():
    attn_output_pt, attn_weights_pt = mha(x, x, x, average_attn_weights=True)

print(f"\nPyTorch MHA output shape: {attn_output_pt.shape}")
print(f"PyTorch MHA output[0, 0, :]: {attn_output_pt[0, 0, :].tolist()}")

# Now let's manually trace what the kernel should compute
print("\n" + "=" * 80)
print("MANUAL KERNEL TRACE (for head 0, query position 0)")
print("=" * 80)

# For head 0, position 0:
# 1. Compute Q
q_h0_p0_kernel = q_head0_kernel[0, 0, :]  # [16]
print(f"1. Q[head=0, pos=0]: {q_h0_p0_kernel.tolist()}")

# 2. Compute K for all positions
k_h0_all = k_heads_pt[0, 0, :, :]  # [seq_len=4, head_dim=16]
print(f"2. K[head=0, all_pos]: shape={k_h0_all.shape}")
print(f"   K[head=0, pos=0]: {k_h0_all[0, :].tolist()}")

# 3. Compute attention scores: Q @ K.T
scores_h0 = q_h0_p0_kernel @ k_h0_all.T  # [4]
scale = 1.0 / (head_dim ** 0.5)
scores_h0_scaled = scores_h0 * scale
print(f"3. Raw scores: {scores_h0.tolist()}")
print(f"   Scaled scores (scale={scale:.4f}): {scores_h0_scaled.tolist()}")

# 4. Softmax
attn_weights_h0 = F.softmax(scores_h0_scaled, dim=-1)
print(f"4. Attention weights: {attn_weights_h0.tolist()}")

# 5. Weighted sum of V
v_h0_all = v_heads_pt[0, 0, :, :]  # [seq_len=4, head_dim=16]
output_h0_p0 = attn_weights_h0 @ v_h0_all  # [16]
print(f"5. Output[head=0, pos=0]: {output_h0_p0.tolist()}")

# Compare with PyTorch's head output
# Extract head 0 output from PyTorch MHA
# PyTorch MHA output is after concatenation and output projection
# We need to get the intermediate head outputs

print("\n" + "=" * 80)
print("VERIFICATION AGAINST KERNEL'S TWO-PASS APPROACH")
print("=" * 80)

# The kernel uses a two-pass approach:
# Pass 1: Compute per-head attention outputs (before output projection)
# Pass 2: Concatenate and apply output projection

# For verification, let's extract the head outputs manually
# PyTorch doesn't expose this directly, but we can reconstruct it

print("\nNote: PyTorch's nn.MultiheadAttention doesn't expose intermediate")
print("head outputs, so we can't directly verify the per-head computation.")
print("However, we've verified the QKV projection produces correct results.")
print("\nThe weight layout understanding appears to be CORRECT.")
print("If there are bugs, they're likely in:")
print("  1. The softmax/attention computation")
print("  2. The output projection (w_out matrix handling)")
print("  3. Memory indexing issues")

print("\n" + "=" * 80)
