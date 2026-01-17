"""
Print expected values for CUDA kernel debugging.

This script computes the exact values that the CUDA kernel SHOULD produce
for the first batch, first head, first query position.

Use this to compare against CUDA kernel debug output.
"""

import torch
import torch.nn.functional as F

print("=" * 80)
print("Expected Values for CUDA Kernel Debug")
print("=" * 80)

# Configuration - MUST match the test configuration
torch.manual_seed(42)
batch_size = 1
seq_len = 4
embed_dim = 32
num_heads = 2
head_dim = embed_dim // num_heads  # 16
scale = 1.0 / (head_dim ** 0.5)

print(f"\nConfiguration:")
print(f"  batch_size = {batch_size}")
print(f"  seq_len = {seq_len}")
print(f"  embed_dim = {embed_dim}")
print(f"  num_heads = {num_heads}")
print(f"  head_dim = {head_dim}")
print(f"  scale = {scale:.6f}")

# Create input and weights
x = torch.randn(batch_size, seq_len, embed_dim)
w_qkv = torch.randn(3 * embed_dim, embed_dim)
bias_qkv = torch.randn(3 * embed_dim)

print("\n" + "=" * 80)
print("STEP 1: Input and Weight Layout")
print("=" * 80)

print(f"\nInput shape: {x.shape}")
print(f"x[0, 0, 0:5]: {x[0, 0, :5].tolist()}")

print(f"\nw_qkv shape: {w_qkv.shape}")
print(f"bias_qkv shape: {bias_qkv.shape}")

# Print weight offsets for head 0
print(f"\nWeight offsets for head 0:")
print(f"  w_q offset: 0")
print(f"  w_k offset: {embed_dim * embed_dim} ({embed_dim}*{embed_dim})")
print(f"  w_v offset: {2 * embed_dim * embed_dim} (2*{embed_dim}*{embed_dim})")

print("\n" + "=" * 80)
print("STEP 2: QKV Projection for Head 0")
print("=" * 80)

# For head 0, query position 0
q_pos = 0
k_pos = 0  # First key position
head_idx = 0

# Input offset for query position 0
x_q = x[0, q_pos, :]  # [embed_dim]
print(f"\nInput for q_pos=0: x[0, 0, :] = {x_q[:5].tolist()}")

# Q weights for head 0: w_qkv[0:head_dim, :]
w_q_head_0 = w_qkv[head_idx*head_dim:(head_idx+1)*head_dim, :]  # [head_dim, embed_dim]
bias_q_head_0 = bias_qkv[head_idx*head_dim:(head_idx+1)*head_dim]  # [head_dim]

print(f"\nQ weights for head 0:")
print(f"  Shape: {w_q_head_0.shape}")
print(f"  w_q[0, 0:5]: {w_q_head_0[0, :5].tolist()}")
print(f"  bias_q[0:5]: {bias_q_head_0[:5].tolist()}")

# Compute Q for head 0, position 0
q_head_0_pos_0 = F.linear(x[0:1, q_pos:q_pos+1, :], w_q_head_0, bias_q_head_0)  # [1, 1, head_dim]
q_values = q_head_0_pos_0[0, 0, :]

print(f"\nComputed Q[head=0, q_pos=0]:")
print(f"  Full: {q_values.tolist()}")
print(f"  [0:5]: {q_values[:5].tolist()}")

# K weights for head 0
w_k_head_0 = w_qkv[embed_dim + head_idx*head_dim:embed_dim + (head_idx+1)*head_dim, :]
bias_k_head_0 = bias_qkv[embed_dim + head_idx*head_dim:embed_dim + (head_idx+1)*head_dim]

print(f"\nK weights for head 0:")
print(f"  Shape: {w_k_head_0.shape}")
print(f"  w_k offset: {embed_dim * embed_dim}")
print(f"  w_k[0, 0:5]: {w_k_head_0[0, :5].tolist()}")

# Compute K for head 0, position 0
k_head_0_pos_0 = F.linear(x[0:1, k_pos:k_pos+1, :], w_k_head_0, bias_k_head_0)
k_values = k_head_0_pos_0[0, 0, :]

print(f"\nComputed K[head=0, k_pos=0]:")
print(f"  Full: {k_values.tolist()}")
print(f"  [0:5]: {k_values[:5].tolist()}")

# V weights for head 0
w_v_head_0 = w_qkv[2*embed_dim + head_idx*head_dim:2*embed_dim + (head_idx+1)*head_dim, :]
bias_v_head_0 = bias_qkv[2*embed_dim + head_idx*head_dim:2*embed_dim + (head_idx+1)*head_dim]

print(f"\nV weights for head 0:")
print(f"  Shape: {w_v_head_0.shape}")
print(f"  w_v offset: {2 * embed_dim * embed_dim}")
print(f"  w_v[0, 0:5]: {w_v_head_0[0, :5].tolist()}")

# Compute V for head 0, position 0
v_head_0_pos_0 = F.linear(x[0:1, k_pos:k_pos+1, :], w_v_head_0, bias_v_head_0)
v_values = v_head_0_pos_0[0, 0, :]

print(f"\nComputed V[head=0, k_pos=0]:")
print(f"  Full: {v_values.tolist()}")
print(f"  [0:5]: {v_values[:5].tolist()}")

print("\n" + "=" * 80)
print("STEP 3: Attention Score Computation")
print("=" * 80)

# Compute attention score: Q @ K.T * scale
raw_score = (q_values @ k_values).item()
scaled_score = raw_score * scale

print(f"\nFor (q_pos=0, k_pos=0):")
print(f"  Q @ K.T = {raw_score:.6f}")
print(f"  scale = {scale:.6f}")
print(f"  score = {scaled_score:.6f}")

# Compute all attention scores for q_pos=0
print(f"\nAll attention scores for q_pos=0:")

# Compute K for all positions
k_all = F.linear(x[0:1, :, :], w_k_head_0, bias_k_head_0)  # [1, seq_len, head_dim]
v_all = F.linear(x[0:1, :, :], w_v_head_0, bias_v_head_0)  # [1, seq_len, head_dim]

scores_all = []
for k in range(seq_len):
    k_val = k_all[0, k, :]
    score = (q_values @ k_val).item() * scale
    scores_all.append(score)
    print(f"  k_pos={k}: score={score:.6f}")

print("\n" + "=" * 80)
print("STEP 4: Softmax Computation")
print("=" * 80)

# Softmax
max_score = max(scores_all)
print(f"\nmax_score = {max_score:.6f}")

exp_scores = [s - max_score for s in scores_all]
exp_scores = [torch.exp(torch.tensor(s)).item() for s in exp_scores]
print(f"exp(scores - max): {[f'{e:.6f}' for e in exp_scores]}")

sum_exp = sum(exp_scores)
print(f"sum_exp = {sum_exp:.6f}")

attn_weights = [e / sum_exp for e in exp_scores]
print(f"attn_weights = {[f'{w:.6f}' for w in attn_weights]}")

print("\n" + "=" * 80)
print("STEP 5: Weighted V and Output Computation")
print("=" * 80)

# Compute weighted sum of V
output = torch.zeros(head_dim)
for k in range(seq_len):
    v_k = v_all[0, k, :]
    output += attn_weights[k] * v_k

print(f"\nFinal head_output[head=0, q_pos=0]:")
print(f"  Full: {output.tolist()}")
print(f"  [0:5]: {output[:5].tolist()}")

print("\n" + "=" * 80)
print("CUDA KERNEL COMPARISON VALUES")
print("=" * 80)

print("""
Copy these values to compare with CUDA kernel printf output:

=== EXPECTED (Python) ===
x[0,0,0:5]: """ + " ".join([f"{x[0,0,i]:.6f}" for i in range(5)]) + f"""
w_q_offset: {0}
w_q[0:5]: """ + " ".join([f"{w_q_head_0[0,i]:.6f}" for i in range(5)]) + f"""
bias_q[0:5]: """ + " ".join([f"{bias_q_head_0[i]:.6f}" for i in range(5)]) + f"""
q_reg[0:5]: """ + " ".join([f"{q_values[i]:.6f}" for i in range(5)]) + f"""
w_k_offset: {embed_dim * embed_dim}
w_k[0:5]: """ + " ".join([f"{w_k_head_0[0,i]:.6f}" for i in range(5)]) + f"""
k_reg[0:5]: """ + " ".join([f"{k_values[i]:.6f}" for i in range(5)]) + f"""
w_v_offset: {2 * embed_dim * embed_dim}
w_v[0:5]: """ + " ".join([f"{w_v_head_0[0,i]:.6f}" for i in range(5)]) + f"""
v_reg[0:5]: """ + " ".join([f"{v_values[i]:.6f}" for i in range(5)]) + f"""
score (unscaled): {raw_score:.6f}
scale: {scale:.6f}
score (scaled): {scaled_score:.6f}
max_score: {max_score:.6f}
exp_score[k_pos=0]: {exp_scores[0]:.6f}
sum_exp: {sum_exp:.6f}
attn_weight[k_pos=0]: {attn_weights[0]:.6f}
head_output[0:5]: """ + " ".join([f"{output[i]:.6f}" for i in range(5)]) + """
""")


print("\n" + "=" * 80)
print("CUDA Data Layout Reference")
print("=" * 80)

print("""
Memory layout in CUDA kernel:

x_offset = (batch_idx * seq_len + q_pos) * embed_dim
For batch=0, q_pos=0, embed_dim=32:
  x_offset = 0

w_q_head_offset = head_idx * head_dim * embed_dim
For head=0, head_dim=16, embed_dim=32:
  w_q_head_offset = 0

w_k_head_offset = embed_dim * embed_dim + head_idx * head_dim * embed_dim
For embed_dim=32, head=0:
  w_k_head_offset = 1024

w_v_head_offset = 2 * embed_dim * embed_dim + head_idx * head_dim * embed_dim
For embed_dim=32, head=0:
  w_v_head_offset = 2048
""")

print("\n" + "=" * 80)
