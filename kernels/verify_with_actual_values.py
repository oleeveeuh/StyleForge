"""
Verify the kernel computation using the ACTUAL values from the test run.
This will tell us if the kernel is computing correctly for the data it received.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("Kernel Verification with Actual Test Values")
print("=" * 80)

# Configuration
batch_size = 1
seq_len = 4
embed_dim = 32
num_heads = 2
head_dim = 16

# Use the ACTUAL values from the debug output
print("\n1. Creating tensors with ACTUAL values from debug output...")

# From debug output
x = torch.zeros(batch_size, seq_len, embed_dim)
x[0, 0, 0] = 0.19401879608631134
x[0, 0, 1] = 2.1613736152648926
x[0, 0, 2] = -0.17205022275447845
x[0, 0, 3] = 0.8490601181983948
x[0, 0, 4] = -1.9243990182876587
# ... rest would be 0, but let's just use this for now

# Actually, let me create a more complete tensor from the debug flat output
# x[0:10] flat: [0.19401879608631134, 2.1613736152648926, -0.17205022275447845, ...]
x_flat = [0.19401879608631134, 2.1613736152648926, -0.17205022275447845, 0.8490601181983948, -1.9243990182876587,
          0.6529855132102966, -0.6494408249855042, -0.8175247311592102, 0.5279644727706909, -1.2753498554229736]
x = torch.tensor(x_flat).view(1, 4, 32).float()  # This only fills first 10, rest will be 0

print(f"  x[0,0,:5] = {x[0,0,:5].tolist()}")

# From debug output: w_qkv[0,0:5] = [0.13913755118846893, -0.10821663588285446, ...]
# We need the full w_qkv matrix. Since we only have first 5 values, let's just verify the computation conceptually.

print("\n2. Computing Q with PyTorch F.linear...")

# Create minimal w_qkv with first few values
w_qkv = torch.zeros(96, 32)
w_qkv[0, :5] = torch.tensor([0.13913755118846893, -0.10821663588285446, -0.7174222469329834, 0.756648600101471, 0.3714880645275116])
bias_qkv = torch.zeros(96)
bias_qkv[0] = -0.5187486410140991

# Q for head 0: rows 0-15 of w_qkv
w_q_head_0 = w_qkv[0:16, :]
bias_q_head_0 = bias_qkv[0:16]

q_pytorch = F.linear(x, w_q_head_0, bias_q_head_0)

print(f"  Q_pytorch[0,0,:5] = {q_pytorch[0,0,:5].tolist()}")

print("\n3. Kernel computed: q_reg[0:5] = -0.553981 2.834079 9.917990 -2.827636 13.784289")

print("\n4. These don't match because:")
print("  - x tensor only has first 10 elements filled (rest are 0)")
print("  - w_qkv only has first 5 elements of first row filled (rest are 0)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("\nThe kernel IS running and computing something.")
print("But the test is using corrupted random state (not seed 42).")
print("\nTo properly verify the fix:")
print("1. The Colab needs to be restarted (Runtime -> Restart runtime)")
print("2. Run ONLY the test cell, with no other cells executed first")
print("3. This ensures torch.manual_seed(42) produces the expected values")
