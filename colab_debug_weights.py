# ============================================
# DETAILED DEBUG: WEIGHT LAYOUT TRACE
# ============================================

import torch
import torch.nn as nn
import sys

print("=" * 70)
print("  DETAILED WEIGHT LAYOUT DEBUG")
print("=" * 70)

# Small test configuration
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
x = torch.randn(batch_size, seq_len, embed_dim)
w_qkv = torch.randn(3 * embed_dim, embed_dim)
w_out = torch.randn(embed_dim, embed_dim)

# PyTorch reference
attn_pt = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
attn_pt.eval()

with torch.no_grad():
    attn_pt.in_proj_weight.copy_(w_qkv)
    attn_pt.out_proj.weight.copy_(w_out)
    out_pt, _ = attn_pt(x, x, x)

print(f"\nPyTorch output shape: {out_pt.shape}")
print(f"PyTorch output:\n{out_pt}")

# Now let's trace through the kernel computation manually
print("\n" + "=" * 70)
print("  MANUAL KERNEL COMPUTATION TRACE")
print("=" * 70)

# Kernel computes Q, K, V for each head
def qkv_projection_manual(x, w, head_idx, head_dim, embed_dim):
    """Compute Q, K, or V for a specific head"""
    # w shape: [3*embed_dim, embed_dim]
    # For head h, Q uses rows [h*head_dim:(h+1)*head_dim]
    w_head = w[head_idx * head_dim:(head_idx + 1) * head_dim, :]  # [head_dim, embed_dim]
    # x @ w_head.T gives [batch, seq, head_dim]
    return x @ w_head.T

# Test QKV projection for first batch, first position
x_00 = x[0, 0, :]  # [embed_dim]

for head in range(num_heads):
    print(f"\n--- Head {head} ---")

    # Q weights for this head
    w_q_head_offset = head * head_dim * embed_dim
    w_q_head = w_qkv[head * head_dim:(head + 1) * head_dim, :]
    print(f"  Q weight shape: {w_q_head.shape}")

    # K weights for this head
    w_k_head_offset = embed_dim * embed_dim + head * head_dim * embed_dim
    w_k_head = w_qkv[embed_dim + head * head_dim:embed_dim + (head + 1) * head_dim, :]
    print(f"  K weight shape: {w_k_head.shape}")

    # V weights for this head
    w_v_head_offset = 2 * embed_dim * embed_dim + head * head_dim * embed_dim
    w_v_head = w_qkv[2 * embed_dim + head * head_dim:2 * embed_dim + (head + 1) * head_dim, :]
    print(f"  V weight shape: {w_v_head.shape}")

    # Compute Q, K, V for x[0, 0]
    q = x_00 @ w_q_head.T
    k = x_00 @ w_k_head.T
    v = x_00 @ w_v_head.T

    print(f"  Q[0,0]: {q}")
    print(f"  K[0,0]: {k}")
    print(f"  V[0,0]: {v}")

# Now test output projection
print("\n" + "=" * 70)
print("  OUTPUT PROJECTION CHECK")
print("=" * 70)

# Simulate head outputs (just use identity for now)
head_outputs = torch.randn(num_heads, head_dim)
print(f"\nHead outputs (simulated):")
for h in range(num_heads):
    print(f"  Head {h}: {head_outputs[h]}")

# Concatenate
concat = head_outputs.view(-1)  # [embed_dim]
print(f"\nConcatenated: {concat}")

# Kernel's output projection for out_dim=0
out_dim = 0
total = 0.0
for h in range(num_heads):
    for i in range(head_dim):
        total += head_outputs[h, i] * w_out[out_dim, h * head_dim + i]

print(f"\nKernel output[0] = sum_h head[h] @ w_out[0, h*head_dim:(h+1)*head_dim]")
print(f"Kernel output[0] = {total}")

# PyTorch's computation
out_pt_manual = concat @ w_out.T
print(f"\nPyTorch output[0] = concat @ w_out.T")
print(f"PyTorch output[0] = {out_pt_manual[0]}")

print(f"\nMatch: {abs(total - out_pt_manual[0]) < 1e-6}")

# Now let's verify the entire computation for one position
print("\n" + "=" * 70)
print("  FULL COMPUTATION CHECK FOR ONE POSITION")
print("=" * 70)

# Compute attention manually using PyTorch's formula
# For each position, compute Q, K, V, then attention

# For position 0:
x_pos0 = x[0, 0, :]  # [embed_dim]

# Q, K, V for all heads
Q = torch.zeros(num_heads, head_dim)
K = torch.zeros(num_heads, head_dim)
V = torch.zeros(num_heads, head_dim)

for h in range(num_heads):
    w_q = w_qkv[h * head_dim:(h + 1) * head_dim, :]
    w_k = w_qkv[embed_dim + h * head_dim:embed_dim + (h + 1) * head_dim, :]
    w_v = w_qkv[2 * embed_dim + h * head_dim:2 * embed_dim + (h + 1) * head_dim, :]

    Q[h] = x_pos0 @ w_q.T
    K[h] = x_pos0 @ w_k.T
    V[h] = x_pos0 @ w_v.T

print(f"\nQ for position 0:")
print(Q)
print(f"\nK for position 0:")
print(K)
print(f"\nV for position 0:")
print(V)

# Compute attention weights for each head
scale = (embed_dim // num_heads) ** -0.5
print(f"\nScale: {scale}")

head_outputs_manual = torch.zeros(num_heads, head_dim)

for h in range(num_heads):
    # Attention scores for this head
    scores = torch.zeros(seq_len)
    for t in range(seq_len):
        # K at position t
        x_t = x[0, t, :]
        w_k = w_qkv[embed_dim + h * head_dim:embed_dim + (h + 1) * head_dim, :]
        k_t = x_t @ w_k.T
        scores[t] = (Q[h] * k_t).sum() * scale

    # Softmax
    scores_exp = torch.exp(scores - scores.max())
    attn_weights = scores_exp / scores_exp.sum()

    print(f"\nHead {h} attention scores: {scores}")
    print(f"Head {h} attention weights: {attn_weights}")

    # Weighted sum of V
    v_output = torch.zeros(head_dim)
    for t in range(seq_len):
        x_t = x[0, t, :]
        w_v = w_qkv[2 * embed_dim + h * head_dim:2 * embed_dim + (h + 1) * head_dim, :]
        v_t = x_t @ w_v.T
        v_output += attn_weights[t] * v_t

    head_outputs_manual[h] = v_output
    print(f"Head {h} output: {v_output}")

# Concatenate and project
concat = head_outputs_manual.view(-1)  # [embed_dim]
print(f"\nConcatenated head outputs: {concat}")

# Output projection
output_manual = concat @ w_out.T
print(f"Manual output[0]: {output_manual}")
print(f"PyTorch output[0]: {out_pt[0, 0, :]}")
print(f"Match: {torch.allclose(output_manual, out_pt[0, 0, :], atol=1e-5)}")

print("\n" + "=" * 70)
