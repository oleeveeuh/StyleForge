# CUDA Kernel Debug Guide

## Overview

The CUDA kernel now has comprehensive debug prints that will output intermediate values for the first batch, first head, first query position, and first key position. This allows direct comparison between the CUDA kernel's computation and the expected Python values.

## Debug Tools

### 1. Expected Values Script

**File:** `kernels/print_expected_values.py`

Run this script to compute the expected values using pure PyTorch:

```bash
python3 kernels/print_expected_values.py
```

This will output:
- Input values
- Q, K, V weight matrix values
- Computed Q, K, V projections
- Attention scores
- Softmax values
- Final head output

### 2. CUDA Kernel Debug Output

The kernel now prints debug information when:
- `batch_idx == 0`
- `head_idx == 0`
- `q_pos == 0`
- `k_pos == 0`

The debug output includes:
```
=== KERNEL DEBUG: batch=0, head=0, q_pos=0, k_pos=0 ===
embed_dim=32, HEAD_DIM=16, head_idx=0, scale=0.250000
x_offset=0, w_q_head_offset=0
Input x[x_offset:x_offset+5]: ...
w_qkv[w_q_head_offset + 0:5]: ...
bias_q[0:5]: ...
q_reg[0:5]: ...
...
raw_score (Q.K^T): ...
scaled_score: ...
max_score: ...
exp_score (k_pos=0): ...
sum_exp: ...
attn_weight (k_pos=0): ...
Final head_output[head=0, q_pos=0, 0:5]: ...
```

### 3. Run the Kernel with Debug

**Option A: Standalone Script**
```bash
python3 kernels/debug_cuda_kernel.py
```

**Option B: Colab/Notebook**
```python
# Run the Colab debug script
%run kernels/debug_kernel_colab.py
```

**Option C: In your own code**
```python
import torch
from styleforge import fused_attention_v1

torch.manual_seed(42)
x = torch.randn(1, 4, 32, device='cuda')
w_qkv = torch.randn(96, 32, device='cuda')
# ... set up other tensors ...

# The kernel will print debug output to stdout
with torch.no_grad():
    output = fused_attention_v1(x, w_qkv, w_out, bias_qkv, bias_out, scale, 2)
```

## Comparing Output

1. Run `print_expected_values.py` to get expected Python values
2. Run the CUDA kernel to get actual CUDA values
3. Compare the two outputs side-by-side

Look for discrepancies in:
- **Input x values** - should match exactly
- **Weight values (w_qkv)** - should match exactly
- **Bias values** - should match exactly
- **Computed Q values (q_reg)** - first divergence point indicates QKV projection bug
- **Computed K values (k_reg)** - indicates K projection bug
- **Computed V values (v_reg)** - indicates V projection bug
- **Attention score** - indicates dot product or scale bug
- **max_score, exp_score, sum_exp** - indicates softmax/reduction bug
- **Final head_output** - indicates output reduction bug

## Common Issues

### Issue: Q values don't match

**Possible causes:**
- Weight offset calculation wrong
- Bias offset calculation wrong
- Vectorized load bug

**Check:**
- `w_q_head_offset = head_idx * head_dim * embed_dim`
- For head=0, this should be 0

### Issue: K values don't match

**Possible causes:**
- K weight offset wrong
- Should be: `embed_dim * embed_dim + head_idx * head_dim * embed_dim`

**Check:**
- For embed_dim=32, head=0: offset should be 1024

### Issue: V values don't match

**Possible causes:**
- V weight offset wrong
- Should be: `2 * embed_dim * embed_dim + head_idx * head_dim * embed_dim`

**Check:**
- For embed_dim=32, head=0: offset should be 2048

### Issue: Attention score doesn't match

**Possible causes:**
- Dot product computation wrong
- Scale factor wrong

**Check:**
- `score = sum(q_reg[i] * k_reg[i]) * scale`

### Issue: Softmax values don't match

**Possible causes:**
- Max reduction bug
- Sum reduction bug
- Thread synchronization issue

### Issue: Final output doesn't match

**Possible causes:**
- V accumulation bug
- Warp reduction bug
- Output write offset wrong

## Removing Debug Prints

Once debugging is complete, remove the debug prints by:

1. Search for `// DEBUG:` comments in `attention.cu`
2. Remove the associated `printf` statements and `if` conditions

Or keep them disabled with a runtime flag:
```cpp
// At top of kernel
const bool DEBUG_PRINT = false;

// Replace debug conditions
if (DEBUG_PRINT && batch_idx == 0 && head_idx == 0 && ...)
```

## Expected Values Reference

For `torch.manual_seed(42)`, `batch_size=1`, `seq_len=4`, `embed_dim=32`, `num_heads=2`:

```
x[0,0,0:5]: 1.926915 1.487284 0.900717 -2.105521 0.678418
w_q_offset: 0
w_q[0,0:5]: 1.931161 1.011864 -1.436406 -1.129860 -0.136035
bias_q[0:5]: 0.070630 -0.068063 1.269279 2.291021 -0.079681
q_reg[0:5]: 4.962327 -1.177215 2.745834 7.978457 5.125279
w_k_offset: 1024
w_k[0,0:5]: -1.407819 -0.080111 0.519412 1.170889 2.177980
k_reg[0:5]: -1.102318 7.333121 2.804938 -14.030498 4.522919
w_v_offset: 2048
```
