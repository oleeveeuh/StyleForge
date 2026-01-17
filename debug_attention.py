"""
Debug script to understand the attention kernel bug.
"""
import torch
import torch.nn as nn

# Small test configuration
batch_size = 1
seq_len = 4
embed_dim = 32
num_heads = 2
head_dim = embed_dim // num_heads

torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, embed_dim, device='cuda', dtype=torch.float32)
w_qkv = torch.randn(3 * embed_dim, embed_dim, device='cuda', dtype=torch.float32)
w_out = torch.randn(embed_dim, embed_dim, device='cuda', dtype=torch.float32)
scale = (embed_dim // num_heads) ** -0.5

print(f"Configuration: batch={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, num_heads={num_heads}, head_dim={head_dim}")
print(f"Scale: {scale}")

# PyTorch reference
attn_pt = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
attn_pt.eval()

with torch.no_grad():
    attn_pt.in_proj_weight.copy_(w_qkv)
    attn_pt.out_proj.weight.copy_(w_out.T)
    out_pt, _ = attn_pt(x, x, x)

print(f"\nPyTorch output shape: {out_pt.shape}")
print(f"PyTorch output[0, 0]: {out_pt[0, 0]}")

# Manual computation to verify
# Split w_qkv into Q, K, V
w_q = w_qkv[:embed_dim, :]      # [embed_dim, embed_dim]
w_k = w_qkv[embed_dim:2*embed_dim, :]
w_v = w_qkv[2*embed_dim:, :]

# For multi-head, split each weight matrix by head
# Head h uses rows h*head_dim to (h+1)*head_dim
print(f"\n=== Manual Multi-Head Attention Computation ===")

# Compute Q, K, V for all heads
Q_all = x @ w_q.T  # [batch, seq, embed_dim]
K_all = x @ w_k.T
V_all = x @ w_v.T

print(f"Q_all shape: {Q_all.shape}")

# Reshape for multi-head: [batch, seq, num_heads, head_dim]
Q_heads = Q_all.view(batch_size, seq_len, num_heads, head_dim)
K_heads = K_all.view(batch_size, seq_len, num_heads, head_dim)
V_heads = V_all.view(batch_size, seq_len, num_heads, head_dim)

print(f"Q_heads shape: {Q_heads.shape}")

# Compute attention for each head
head_outputs = []
for h in range(num_heads):
    Q_h = Q_heads[:, :, h, :]  # [batch, seq, head_dim]
    K_h = K_heads[:, :, h, :]
    V_h = V_heads[:, :, h, :]

    # Attention: Q @ K.T * scale
    attn_scores = (Q_h @ K_h.transpose(-2, -1)) * scale  # [batch, seq, seq]
    attn_weights = torch.softmax(attn_scores, dim=-1)
    head_out = attn_weights @ V_h  # [batch, seq, head_dim]

    print(f"\nHead {h}:")
    print(f"  Q_h[0, 0]: {Q_h[0, 0]}")
    print(f"  K_h[0, 0]: {K_h[0, 0]}")
    print(f"  attn_scores[0, 0]: {attn_scores[0, 0]}")
    print(f"  attn_weights[0, 0]: {attn_weights[0, 0]}")
    print(f"  head_out[0, 0]: {head_out[0, 0]}")

    head_outputs.append(head_out)

# Concatenate heads
concat = torch.cat(head_outputs, dim=-1)  # [batch, seq, embed_dim]
print(f"\nConcatenated[0, 0]: {concat[0, 0]}")

# Output projection
out_manual = concat @ w_out.T  # [batch, seq, embed_dim]
print(f"Manual output[0, 0]: {out_manual[0, 0]}")
print(f"PyTorch output[0, 0]: {out_pt[0, 0]}")
print(f"Max difference: {(out_manual - out_pt).abs().max().item():.2e}")

# Now let's verify the kernel is receiving the same inputs
print(f"\n=== Weight Layout Verification ===")
print(f"w_qkv shape: {w_qkv.shape}")
print(f"w_q[0:head_dim, 0:4] (head 0 Q weights first 4 cols): {w_q[0:head_dim, 0:4]}")
print(f"w_q[head_dim:2*head_dim, 0:4] (head 1 Q weights first 4 cols): {w_q[head_dim:2*head_dim, 0:4]}")
