# Attention Kernel Documentation

## Overview

The custom attention kernel implements an optimized multi-head attention with **online softmax** that achieves O(N) memory complexity instead of O(N²) for standard attention.

## Key Innovation: Online Softmax

### Standard Attention (O(N²) memory)

```
1. Compute attention scores: S = Q @ K^T  # [batch, heads, seq, seq]
2. Compute softmax: A = softmax(S)        # [batch, heads, seq, seq]
3. Compute output: O = A @ V              # [batch, heads, seq, head_dim]
```

The attention matrix `S` has size N×N where N is the sequence length. For seq_len=4096, this is 16M elements per head.

### Online Softmax (O(N) memory)

```
For each query position:
1. Initialize: max = -inf, sum_exp = 0, output = 0
2. For each key position:
   a. Compute score = Q[i] @ K[j]
   b. Update max and rescale previous values
   c. Update sum_exp with new contribution
   d. Accumulate weighted value in registers
3. Normalize output by sum_exp
```

No N×N matrix is ever materialized - all computation happens in registers.

## Algorithm Details

### Single-Pass Softmax

The key insight is that softmax can be computed online with proper rescaling:

```cuda
// Online softmax update
float old_max = max_score;
max_score = fmaxf(max_score, new_score);

// Rescale previous accumulator values
float scale_factor = expf(old_max - max_score);
sum_exp = sum_exp * scale_factor + expf(new_score - max_score);

// Apply same rescaling to output accumulation
for (int d = 0; d < HEAD_DIM; d++) {
    v_accum[d] = v_accum[d] * scale_factor + expf(new_score - max_score) * v[d];
}
```

### Warp-Level Reduction

After all threads process their subset of keys, we need to combine results:

```cuda
// Find global maximum across threads
float global_max = warp_reduce_max(thread_max);

// Rescale each thread's contribution
float correction = expf(thread_max - global_max);
sum_exp *= correction;
for (int d = 0; d < HEAD_DIM; d++) {
    v_accum[d] *= correction;
}

// Sum across threads
sum_exp = warp_reduce_sum(sum_exp);
for (int d = 0; d < HEAD_DIM; d++) {
    output[d] = warp_reduce_sum(v_accum[d]) / sum_exp;
}
```

## Performance Characteristics

### Time Complexity

- Standard: O(N²D) for N sequence length, D head dimension
- Online: O(N²D) same asymptotic, but better constants
- Difference: Fewer memory accesses (no attention matrix write/read)

### Space Complexity

- Standard: O(N²) for attention matrix
- Online: O(D) for registers per thread

### Scalability

| Sequence Length | Attention Matrix Size | Memory Saved |
|-----------------|----------------------|--------------|
| 512             | 1 MB per head        | 87.5%        |
| 1024            | 4 MB per head        | 93.8%        |
| 2048            | 16 MB per head       | 96.9%        |
| 4096            | 64 MB per head       | 98.4%        |

## Usage

```python
from kernels.attention_v3_wrapper import FusedAttentionV3

# Create attention layer
attention = FusedAttentionV3(
    embed_dim=4096,    # Model dimension
    num_heads=32,      # Number of attention heads
    bias=False,        # Whether to use bias
).cuda()

# Forward pass
hidden_states = torch.randn(1, 512, 4096).cuda()
output = attention(hidden_states)  # [1, 512, 4096]
```

## Validation

The kernel outputs are validated against PyTorch's `nn.MultiheadAttention`:

```bash
python llm_benchmarks/scripts/validate_attention.py
```

Expected results:
- Max error: < 1e-4
- Mean error: < 1e-6

## References

- Flash Attention: Dao et al. (2022) - https://arxiv.org/abs/2205.14135
- CUDA Warp Primitives: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
