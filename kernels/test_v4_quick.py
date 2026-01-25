"""
Quick test of V4 kernel - direct comparison
"""

import torch
import torch.nn as nn
import numpy as np

print("="*70)
print("Testing V4 Fixed Kernel")
print("="*70)

# Simple test configuration
B, S, E, H = 1, 256, 512, 4
head_dim = E // H

print(f"\nConfig: B={B}, S={S}, E={E}, H={H}, head_dim={head_dim}")

# Create input
x = torch.randn(B, S, E, device='cuda')

# PyTorch baseline
pytorch_attn = nn.MultiheadAttention(E, H, batch_first=True).cuda().eval()

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = pytorch_attn(x, x, x)
torch.cuda.synchronize()

# Benchmark PyTorch
times = []
for _ in range(50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        _ = pytorch_attn(x, x, x)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

pytorch_mean = np.mean(times)
print(f"\nPyTorch MHA: {pytorch_mean:.3f} ms")

# Compute QKV manually for V4
W = torch.randn(3 * E, E, device='cuda')
qkv = x @ W.T  # [B, S, 3*E]
qkv = qkv.reshape(B, S, 3, H, head_dim)
qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, head_dim]
Q_pt, K_pt, V_pt = qkv[0], qkv[1], qkv[2]

# Simple PyTorch attention with pre-computed Q,K,V
scale = 1.0 / (head_dim ** 0.5)
scores = (Q_pt @ K_pt.transpose(-2, -1)) * scale
attn_weights = torch.softmax(scores, dim=-1)
output_pt = attn_weights @ V_pt
output_pt = output_pt.permute(0, 2, 1, 3).reshape(B, S, E)

# Benchmark this
times = []
for _ in range(10):
    with torch.no_grad():
        scores = (Q_pt @ K_pt.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        output_pt = attn_weights @ V_pt
        output_pt = output_pt.permute(0, 2, 1, 3).reshape(B, S, E)
torch.cuda.synchronize()

for _ in range(50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        scores = (Q_pt @ K_pt.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        output_pt = attn_weights @ V_pt
        output_pt = output_pt.permute(0, 2, 1, 3).reshape(B, S, E)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

manual_pt_mean = np.mean(times)
print(f"PyTorch (manual Q,K,V): {manual_pt_mean:.3f} ms")

# Now try V4
try:
    from attention_v4_wrapper import get_attention_v4_module

    module = get_attention_v4_module()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = module.fused_attention_v4(Q_pt, K_pt, V_pt, scale)
    torch.cuda.synchronize()

    # Benchmark V4
    times = []
    for _ in range(50):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = module.fused_attention_v4(Q_pt, K_pt, V_pt, scale)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    v4_mean = np.mean(times)
    print(f"V4 Custom: {v4_mean:.3f} ms")

    speedup = manual_pt_mean / v4_mean
    print(f"\nSpeedup: {speedup:.2f}x")

    if speedup >= 1.0:
        print(" V4 is FASTER than PyTorch!")
    elif speedup >= 0.5:
        print(" V4 is competitive (within 2x)")
    else:
        print(" V4 is still slower - needs more optimization")

    # Verify correctness
    with torch.no_grad():
        v4_output = module.fused_attention_v4(Q_pt, K_pt, V_pt, scale)

    v4_output = v4_output.permute(0, 2, 1, 3).reshape(B, S, E)

    error = (v4_output - output_pt).abs().max().item()
    print(f"\nMax error vs PyTorch: {error:.2e}")

    if error < 1e-4:
        print(" CORRECTNESS: PASS")
    else:
        print(" CORRECTNESS: FAIL - large error")

except Exception as e:
    print(f"\nV4 Error: {e}")
    import traceback
    traceback.print_exc()
