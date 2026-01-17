"""
Step-by-step simulation of the CUDA kernel's computation
to identify exactly where the bug is.

This simulates what the kernel does, step by step, using PyTorch operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("CUDA Kernel Step-by-Step Simulation")
print("=" * 80)

# Configuration
embed_dim = 32
num_heads = 2
head_dim = embed_dim // num_heads  # 16
batch_size = 1
seq_len = 4
scale = 1.0 / (head_dim ** 0.5)

print(f"\nConfiguration:")
print(f"  embed_dim = {embed_dim}")
print(f"  num_heads = {num_heads}")
print(f"  head_dim = {head_dim}")
print(f"  batch_size = {batch_size}")
print(f"  seq_len = {seq_len}")
print(f"  scale = {scale}")

# Create PyTorch MHA
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
in_proj_weight = mha.in_proj_weight  # [3*embed_dim, embed_dim] = [96, 32]
in_proj_bias = mha.in_proj_bias  # [96]
out_proj_weight = mha.out_proj.weight  # [embed_dim, embed_dim] = [32, 32]
out_proj_bias = mha.out_proj.bias  # [32]

print("\n" + "=" * 80)
print("KERNEL STEP 1: Per-head QKV projection")
print("=" * 80)

# The kernel computes per-head QKV separately
# For each head h, it computes:
#   Q_h = x @ w_qkv[h*head_dim:(h+1)*head_dim, :].T + bias[h*head_dim:(h+1)*head_dim]

# Extract Q, K, V weight sections
w_q = in_proj_weight[:embed_dim, :]      # [32, 32]
w_k = in_proj_weight[embed_dim:2*embed_dim, :]  # [32, 32]
w_v = in_proj_weight[2*embed_dim:, :]    # [32, 32]

b_q = in_proj_bias[:embed_dim]
b_k = in_proj_bias[embed_dim:2*embed_dim]
b_v = in_proj_bias[2*embed_dim:]

print(f"\nw_q shape: {w_q.shape}")
print(f"w_k shape: {w_k.shape}")
print(f"w_v shape: {w_v.shape}")

# Simulate KERNEL's per-head projection
head_outputs = []  # Will store [batch, seq_len, head_dim] for each head

for h in range(num_heads):
    # Extract head-specific weights
    w_q_h = w_q[h*head_dim:(h+1)*head_dim, :]  # [head_dim, embed_dim]
    w_k_h = w_k[h*head_dim:(h+1)*head_dim, :]
    w_v_h = w_v[h*head_dim:(h+1)*head_dim, :]

    b_q_h = b_q[h*head_dim:(h+1)*head_dim]
    b_k_h = b_k[h*head_dim:(h+1)*head_dim]
    b_v_h = b_v[h*head_dim:(h+1)*head_dim]

    # Compute Q, K, V for this head
    q_h = F.linear(x, w_q_h, b_q_h)  # [batch, seq_len, head_dim]
    k_h = F.linear(x, w_k_h, b_k_h)
    v_h = F.linear(x, w_v_h, b_v_h)

    print(f"\nHead {h}:")
    print(f"  q_h shape: {q_h.shape}")
    print(f"  q_h[0, 0, :4]: {q_h[0, 0, :4].tolist()}")

    # Store for attention computation
    head_outputs.append({'q': q_h, 'k': k_h, 'v': v_h})

print("\n" + "=" * 80)
print("KERNEL STEP 2: Attention computation (per-head)")
print("=" * 80)

# Compute attention for each head
head_attn_outputs = []  # Will store [batch, seq_len, head_dim] for each head

for h in range(num_heads):
    q_h = head_outputs[h]['q']  # [batch, seq_len, head_dim]
    k_h = head_outputs[h]['k']
    v_h = head_outputs[h]['v']

    # Reshape for attention: [batch, seq_len, head_dim]
    # The kernel processes each position independently

    # For each query position, compute attention over all key positions
    batch_head_output = []

    for b in range(batch_size):
        seq_head_output = []

        for q_pos in range(seq_len):
            # Q at this position: [head_dim]
            q = q_h[b, q_pos, :]  # [head_dim]

            # K, V at all positions: [seq_len, head_dim]
            k_all = k_h[b, :, :]  # [seq_len, head_dim]
            v_all = v_h[b, :, :]  # [seq_len, head_dim]

            # Attention scores: Q @ K.T * scale
            # q: [head_dim], k_all: [seq_len, head_dim]
            # q @ k_all.T: [seq_len]
            scores = (q @ k_all.T) * scale  # [seq_len]

            # Softmax
            attn_weights = F.softmax(scores, dim=-1)  # [seq_len]

            # Weighted sum of V: attn_weights @ v_all
            # attn_weights: [seq_len], v_all: [seq_len, head_dim]
            # Result: [head_dim]
            output = attn_weights @ v_all  # [head_dim]

            seq_head_output.append(output)

        batch_head_output.append(torch.stack(seq_head_output, dim=0))

    # Stack: [batch, seq_len, head_dim]
    head_output = torch.stack(batch_head_output, dim=0)
    head_attn_outputs.append(head_output)

    print(f"\nHead {h} attention output:")
    print(f"  shape: {head_output.shape}")
    print(f"  output[0, 0, :4]: {head_output[0, 0, :4].tolist()}")

print("\n" + "=" * 80)
print("KERNEL STEP 3: Output projection")
print("=" * 80)

print(f"\nout_proj_weight shape: {out_proj_weight.shape}")

# The kernel's output_projection_kernel computes:
# For each (batch, seq, out_dim):
#   output[batch, seq, out_dim] = sum_h head_output_h @ w_out[out_dim, h*head_dim:(h+1)*head_dim]

# Simulate this
output_kernel = torch.zeros(batch_size, seq_len, embed_dim)

for b in range(batch_size):
    for s in range(seq_len):
        for out_dim in range(embed_dim):
            total = 0.0
            for h in range(num_heads):
                # head_output for this head: [batch, seq, head_dim]
                head_out = head_attn_outputs[h][b, s, :]  # [head_dim]

                # w_out slice: w_out[out_dim, h*head_dim:(h+1)*head_dim]
                w_out_slice = out_proj_weight[out_dim, h*head_dim:(h+1)*head_dim]  # [head_dim]

                # Dot product
                partial = (head_out * w_out_slice).sum()
                total += partial

            output_kernel[b, s, out_dim] = total

# Add bias
if out_proj_bias is not None:
    output_kernel += out_proj_bias

print(f"\nKernel-like output shape: {output_kernel.shape}")
print(f"Kernel-like output[0, 0, :8]: {output_kernel[0, 0, :8].tolist()}")

print("\n" + "=" * 80)
print("VERIFICATION: Compute reference output manually")
print("=" * 80)

# Compute the reference output the way PyTorch does
# Step 1: Full QKV projection
qkv_full = F.linear(x, in_proj_weight, in_proj_bias)  # [batch, seq, 3*embed_dim]
q_full, k_full, v_full = qkv_full.chunk(3, dim=-1)

# Step 2: Split into heads
q_heads = q_full.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
k_heads = k_full.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
v_heads = v_full.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

# Step 3: Compute attention per head
head_ref_outputs = []
for h in range(num_heads):
    q_h = q_heads[:, h, :, :]
    k_h = k_heads[:, h, :, :]
    v_h = v_heads[:, h, :, :]

    # Attention: (Q @ K.T) @ V
    attn_scores = (q_h @ k_h.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_scores, dim=-1)
    head_out = attn_weights @ v_h
    head_ref_outputs.append(head_out)

# Step 4: Concatenate
concat = torch.cat(head_ref_outputs, dim=-1)  # [batch, seq, embed_dim]

print(f"\nConcatenated heads[0, 0, :8]: {concat[0, 0, :8].tolist()}")

# Step 5: Output projection
output_ref = F.linear(concat, out_proj_weight, out_proj_bias)

print(f"Reference output[0, 0, :8]: {output_ref[0, 0, :8].tolist()}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"\nPyTorch MHA output[0, 0, :8]: {attn_output_pt[0, 0, :8].tolist()}")
print(f"Reference output[0, 0, :8]: {output_ref[0, 0, :8].tolist()}")
print(f"Kernel-like output[0, 0, :8]: {output_kernel[0, 0, :8].tolist()}")

diff_ref_pt = (output_ref - attn_output_pt).abs().max()
diff_kernel_pt = (output_kernel - attn_output_pt).abs().max()
diff_kernel_ref = (output_kernel - output_ref).abs().max()

print(f"\nMax difference (Reference vs PyTorch): {diff_ref_pt:.2e}")
print(f"Max difference (Kernel-like vs PyTorch): {diff_kernel_pt:.2e}")
print(f"Max difference (Kernel-like vs Reference): {diff_kernel_ref:.2e}")

if diff_ref_pt < 1e-5:
    print("\n✓ Reference matches PyTorch MHA")
else:
    print("\n✗ Reference does NOT match PyTorch MHA")

if diff_kernel_ref < 1e-5:
    print("✓ Kernel-like matches Reference")
else:
    print("✗ Kernel-like does NOT match Reference")

if diff_kernel_pt < 1e-5:
    print("✓ Kernel-like matches PyTorch MHA")
else:
    print("✗ Kernel-like does NOT match PyTorch MHA")

# If there's a difference, let's debug further
if diff_kernel_pt > 1e-5:
    print("\n" + "=" * 80)
    print("DEBUG: Finding the source of the difference")
    print("=" * 80)

    # Check if head outputs match
    print("\nComparing head outputs:")
    for h in range(num_heads):
        diff = (head_attn_outputs[h] - head_ref_outputs[h]).abs().max()
        print(f"  Head {h}: max diff = {diff:.2e}")

    # Check concat
    concat_kernel = torch.cat(head_attn_outputs, dim=-1)
    diff_concat = (concat_kernel - concat).abs().max()
    print(f"\nConcat max diff: {diff_concat:.2e}")

    # Check output projection
    print(f"\nChecking output projection:")
    print(f"  out_proj_weight[0, :16] = {out_proj_weight[0, :16].tolist()}")
    print(f"  out_proj_weight[0, 16:32] = {out_proj_weight[0, 16:].tolist()}")

    # Manual check of first output dimension
    manual_out_0 = 0.0
    for h in range(num_heads):
        head_out = head_attn_outputs[h][0, 0, :]
        w_out_slice = out_proj_weight[0, h*head_dim:(h+1)*head_dim]
        partial = (head_out * w_out_slice).sum()
        print(f"  Head {h} partial: {partial:.6f}")
        manual_out_0 += partial

    if out_proj_bias is not None:
        manual_out_0 += out_proj_bias[0]

    print(f"  Manual out[0,0,0]: {manual_out_0:.6f}")
    print(f"  Kernel out[0,0,0]: {output_kernel[0, 0, 0]:.6f}")
    print(f"  Reference out[0,0,0]: {output_ref[0, 0, 0]:.6f}")
    print(f"  PyTorch out[0,0,0]: {attn_output_pt[0, 0, 0]:.6f}")

print("\n" + "=" * 80)
