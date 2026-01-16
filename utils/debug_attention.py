"""
Debug script for attention kernel - prints intermediate values
to identify where the computation diverges from PyTorch.
"""

import torch
import torch.nn as nn


def debug_attention_comparison():
    """
    Compare StyleForge attention with PyTorch step by step.
    """
    print("=" * 70)
    print("DEBUG: Attention Kernel Comparison")
    print("=" * 70)

    # Configuration - small for easier debugging
    batch_size = 1
    seq_len = 8  # Small sequence
    embed_dim = 16  # Small embed dim
    num_heads = 2
    head_dim = embed_dim // num_heads

    print(f"\nConfiguration:")
    print(f"  batch_size = {batch_size}")
    print(f"  seq_len = {seq_len}")
    print(f"  embed_dim = {embed_dim}")
    print(f"  num_heads = {num_heads}")
    print(f"  head_dim = {head_dim}")

    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')

    # PyTorch attention
    attn_pytorch = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
    attn_pytorch.eval()

    with torch.no_grad():
        out_pytorch, attn_weights_pytorch = attn_pytorch(x, x, x, average_attn_weights=True)

    print(f"\nPyTorch output shape: {out_pytorch.shape}")
    print(f"PyTorch attention weights shape: {attn_weights_pytorch.shape}")
    print(f"PyTorch output (first 5 values): {out_pytorch[0, 0, :5]}")

    # Now let's compute the expected Q, K, V projections
    print("\n" + "-" * 70)
    print("Expected QKV projections (from PyTorch weights)")
    print("-" * 70)

    # PyTorch's in_proj_weight: [3*embed_dim, embed_dim]
    w_q_pt = attn_pytorch.in_proj_weight[:embed_dim]  # [embed_dim, embed_dim]
    w_k_pt = attn_pytorch.in_proj_weight[embed_dim:2*embed_dim]
    w_v_pt = attn_pytorch.in_proj_weight[2*embed_dim:]

    print(f"Q weight shape: {w_q_pt.shape}")
    print(f"K weight shape: {w_k_pt.shape}")
    print(f"V weight shape: {w_v_pt.shape}")

    # Compute Q, K, V
    x_flat = x.view(batch_size * seq_len, embed_dim)  # [batch*seq, embed_dim]
    q_all = x_flat @ w_q_pt.T  # [batch*seq, embed_dim]
    k_all = x_flat @ w_k_pt.T
    v_all = x_flat @ w_v_pt.T

    # Reshape for multi-head: [batch, num_heads, seq_len, head_dim]
    q = q_all.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k_all.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v_all.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    print(f"\nQ shape after reshape: {q.shape}")
    print(f"K shape after reshape: {k.shape}")
    print(f"V shape after reshape: {v.shape}")

    # Print head 0, query position 0, key position 0
    print(f"\nHead 0, Query 0: {q[0, 0, 0]}")
    print(f"Head 0, Key 0:   {k[0, 0, 0]}")
    print(f"Head 0, Value 0: {v[0, 0, 0]}")

    # Compute attention scores manually
    scale = head_dim ** -0.5
    print(f"\nScale factor: {scale}")

    # Q @ K^T for head 0, query 0
    scores_0 = (q[0, 0, 0] * k[0, 0]).sum(dim=-1) * scale
    print(f"\nScores for head 0, query 0 (before softmax): {scores_0}")

    # Softmax
    attn_weights_0 = torch.softmax(scores_0, dim=-1)
    print(f"Attention weights for head 0, query 0: {attn_weights_0}")

    # Compare with PyTorch's attention weights
    print(f"\nPyTorch attention weights[0, 0, 0]: {attn_weights_pytorch[0, 0, 0]}")

    # Compute weighted sum of V
    v_0 = v[0, 0]  # [seq_len, head_dim]
    output_0 = (attn_weights_0.unsqueeze(-1) * v_0).sum(dim=0)
    print(f"\nComputed output for head 0, query 0: {output_0}")

    # Now try StyleForge
    print("\n" + "-" * 70)
    print("StyleForge Attention")
    print("-" * 70)

    try:
        from kernels import FusedAttention

        attn_fused = FusedAttention(embed_dim, num_heads).cuda()

        # Copy PyTorch weights
        with torch.no_grad():
            attn_fused.w_qkv.copy_(torch.cat([
                w_q_pt, w_k_pt, w_v_pt
            ], dim=0))
            attn_fused.w_out.copy_(attn_pytorch.out_proj.weight.T)
            if attn_pytorch.out_proj.bias is not None and attn_fused.bias_out is not None:
                attn_fused.bias_out.copy_(attn_pytorch.out_proj.bias)

        with torch.no_grad():
            out_fused = attn_fused(x)

        print(f"StyleForge output shape: {out_fused.shape}")
        print(f"StyleForge output (first 5 values): {out_fused[0, 0, :5]}")

        # Compare
        diff = (out_fused - out_pytorch).abs()
        print(f"\nMax difference:  {diff.max().item():.2e}")
        print(f"Mean difference: {diff.mean().item():.2e}")

        # Show element-wise differences for first query
        print(f"\nElement-wise differences for batch 0, seq 0:")
        for i in range(min(5, embed_dim)):
            print(f"  dim {i}: PyTorch={out_pytorch[0,0,i].item():.4f}, "
                  f"Fused={out_fused[0,0,i].item():.4f}, "
                  f"diff={diff[0,0,i].item():.4f}")

    except Exception as e:
        print(f"StyleForge failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)


if __name__ == "__main__":
    debug_attention_comparison()
