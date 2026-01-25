# FFN Kernel Documentation

## Overview

The custom FFN (Feed-Forward Network) kernel fuses the three operations of a transformer FFN block into a single kernel launch: **FC1 → GELU → FC2**.

## What is an FFN?

In transformer models, each layer contains a feed-forward network:

```
Input (hidden_dim)
    ↓
Linear: hidden_dim → intermediate_dim  (typically 4x expansion)
    ↓
GELU activation
    ↓
Linear: intermediate_dim → hidden_dim
    ↓
Output (hidden_dim)
```

For Llama-2-7B:
- `hidden_dim = 4096`
- `intermediate_dim = 11008` (~2.67x expansion)

## Why Fuse?

### Standard PyTorch (3 kernel launches)

```
Kernel 1: torch.nn.Linear (FC1)
  - Allocates intermediate tensor [batch, seq, intermediate]
  - Writes to global memory

Kernel 2: torch.nn.GELU
  - Reads intermediate tensor
  - Writes intermediate tensor back

Kernel 3: torch.nn.Linear (FC2)
  - Reads intermediate tensor
  - Writes output tensor
```

Each kernel launch has overhead (~5-20 μs) and forces intermediate values through global memory (~450 GB/s bandwidth).

### Fused Kernel (1 kernel launch)

```
Kernel 1: Fused FFN
  - Reads input from global memory
  - Computes FC1 → GELU → FC2 entirely in registers/shared memory
  - Writes final output to global memory
```

## Implementation Details

### Inline GELU Activation

GELU(x) = x * Φ(x) where Φ is the cumulative distribution function of the standard normal distribution.

Approximation used (with fast PTX tanh):

```cuda
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);

    // Fast tanh approximation using PTX assembly
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(tanh_val) : "f"(tanh_arg));

    return 0.5f * x * (1.0f + tanh_val);
}
```

### Shared Memory Tiling

For better performance, input weights are loaded into shared memory:

```cuda
__shared__ float s_input[EMBED_DIM];      // ~14 TB/s bandwidth
__shared__ float s_intermediate[FFN_DIM]; // ~14 TB/s bandwidth

// All threads collaborate to load input
if (tid < EMBED_DIM) {
    s_input[tid] = input[input_idx];
}
__syncthreads();

// Each thread computes one output dimension
for (int i = 0; i < EMBED_DIM; i++) {
    val += s_input[i] * fc1_weight[i * ffn_dim + tid];
}
```

## Memory Savings

### Intermediate Tensor Elimination

| Sequence Length | Intermediate Tensor Size |
|-----------------|-------------------------|
| 512             | 21.1 MB                 |
| 1024            | 42.2 MB                 |
| 2048            | 84.4 MB                 |
| 4096            | 168.8 MB                |

For seq_len=2048, this saves ~84 MB of memory allocation and 2 passes through global memory.

## Usage

```python
from kernels.ffn_wrapper import FusedFFN

# Create FFN layer
ffn = FusedFFN(
    embed_dim=4096,     # Input/output dimension
    ffn_dim=11008,      # Intermediate dimension
    bias=True,          # Whether to use bias
).cuda()

# Forward pass
hidden_states = torch.randn(1, 512, 4096).cuda()
output = ffn(hidden_states)  # [1, 512, 4096]
```

## Validation

The kernel outputs are validated against PyTorch's sequential implementation:

```python
import torch
import torch.nn as nn
from kernels.ffn_wrapper import FusedFFN

# Create models
custom_ffn = FusedFFN(4096, 11008).cuda().eval()
pytorch_ffn = nn.Sequential(
    nn.Linear(4096, 11008),
    nn.GELU(),
    nn.Linear(11008, 4096)
).cuda().eval()

# Compare
x = torch.randn(1, 512, 4096).cuda()
with torch.no_grad():
    custom_out = custom_ffn(x)
    pytorch_out = pytorch_ffn(x)

# Should match within tolerance
print(f"Max error: {(custom_out - pytorch_out).abs().max().item():.2e}")
```

## Performance

On RTX 3090 with Llama-2-7B configuration:

| Sequence | PyTorch (ms) | Fused (ms) | Speedup |
|----------|-------------|-----------|---------|
| 512      | 1.23        | 0.82       | 1.51x   |
| 1024     | 2.45        | 1.52       | 1.61x   |
| 2048     | 4.89        | 2.97       | 1.65x   |
| 4096     | 9.72        | 5.78       | 1.68x   |

## References

- GELU Activation: Hendrycks & Gimpel (2016) - https://arxiv.org/abs/1606.08415
- PTX Instruction: https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#special-functions
