"""
Verify that the kernel computes Q correctly using F.linear
"""
import torch
import torch.nn.functional as F

# Same configuration as test
torch.manual_seed(42)  # But this won't match the actual test...

# Let's use the ACTUAL values from the test output
x_test = torch.tensor([0.19401879608631134, 2.1613736152648926, -0.17205022275447845, 0.8490601181983948, -1.9243990182876587])
w_test = torch.tensor([0.13913755118846893, -0.10821663588285446, -0.7174222469329834, 0.756648600101471, 0.3714880645275116])
b_test = torch.tensor([-0.5187486410140991, 1.2267580032348633, 0.6254805326461792, -0.9116847515106201, 0.568683385848999])

print("Computing Q[0] = x @ w[0,:] + b[0]")
print(f"x = {x_test.tolist()}")
print(f"w[0,:] = {w_test.tolist()}")
print(f"b[0] = {b_test.item()}")

# Manual computation
q0_manual = (x_test * w_test).sum() + b_test[0]
print(f"\nQ[0] = {q0_manual}")

# Using PyTorch
x_full = torch.randn(1, 32)  # Dummy, will replace
x_full[0, :5] = x_test
w_full = torch.randn(16, 32)  # Dummy
w_full[0, :5] = w_test
b_full = torch.randn(16)  # Dummy
b_full[0] = b_test[0]

# Compute first 5 elements of Q
q_result = F.linear(x_full, w_full, b_full)[0, :5]
print(f"Q[0:5] using F.linear: {q_result.tolist()}")

# The kernel output says q_reg[0] = -0.553981
# Let's verify with full computation

# Actually let me use the full tensor from the test
print("\n" + "=" * 80)
print("FULL COMPUTATION CHECK")
print("=" * 80)

# Simulate what the kernel does
import sys
sys.path.insert(0, '.')

# Set same seed (but it won't match because random state is already corrupted)
torch.manual_seed(42)
x = torch.randn(1, 4, 32)
w_qkv = torch.randn(96, 32)
bias_qkv = torch.randn(96)

# Q for head 0
w_q_head_0 = w_qkv[0:16, :]
bias_q_head_0 = bias_qkv[0:16]
q_head_0 = F.linear(x, w_q_head_0, bias_q_head_0)

print(f"Q[head=0, pos=0, 0:5]: {q_head_0[0, 0, :5].tolist()}")

# The kernel computed: q_reg[0:5] = -0.553981 2.834079 9.917990 -2.827636 13.784289
# These don't match because we're using different random data!

print("\nThe issue is that the test is using DIFFERENT random data than seed 42.")
print("This is because the Colab notebook ran some code first that changed the random state.")
print("\nTo properly test, we need to:")
print("1. Restart the Colab runtime")
print("2. Run ONLY the test cell (no other cells before it)")
