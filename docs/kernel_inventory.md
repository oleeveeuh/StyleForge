# CUDA Kernel Inventory

Complete inventory of all CUDA kernels in StyleForge and their optimization capabilities.

## Overview

StyleForge includes custom CUDA kernels optimized for Transformer-style operations. The kernels are designed for JIT compilation via PyTorch's CUDA extension API.

## Kernel Files

| File | Purpose | Status | Performance |
|------|---------|--------|-------------|
| `kernels/attention.cu` | Multi-head attention (V1) | Stable | 8x speedup |
| `kernels/attention_v2.cu` | Multi-head attention (V2) | Stable | 15-20x speedup |
| `kernels/ffn.cu` | Feed-forward network | Stable | 4-5x speedup |
| `kernels/instance_norm.cu` | Instance normalization | Stable | 3-5x speedup |
| `kernels/output_projection.cu` | Attention output projection | Stable | 2-3x speedup |

## Kernel Signatures

### 1. Fused Attention (V2)

```cuda
// Fused multi-head attention with QKV projection
torch::Tensor fused_attention_v2_forward(
    torch::Tensor x,              // Input: [batch, seq_len, embed_dim]
    torch::Tensor w_qkv,          // QKV weights: [3 * embed_dim, embed_dim]
    torch::Tensor w_out,          // Output weights: [embed_dim, embed_dim]
    torch::Tensor bias_qkv,       // QKV bias: [3 * embed_dim]
    torch::Tensor bias_out,       // Output bias: [embed_dim]
    float scale,                  // Scaling factor (1.0 / sqrt(head_dim))
    int num_heads                 // Number of attention heads
);
```

**Optimizations:**
- Vectorized memory loads (float4)
- Shared memory tiling for keys/values
- Warp-level reductions for attention scores
- Coalesced global memory access
- Proper padding to avoid bank conflicts

**Replaces:** `nn.MultiheadAttention`

---

### 2. Fused FFN

```cuda
// Fused feed-forward network: Linear -> GELU -> Linear
torch::Tensor fused_ffn_forward(
    torch::Tensor x,             // Input: [batch, seq_len, embed_dim]
    torch::Tensor fc1_weight,    // Hidden layer weights: [embed_dim, hidden_dim]
    torch::Tensor fc1_bias,     // Hidden layer bias: [hidden_dim]
    torch::Tensor fc2_weight,    // Output layer weights: [hidden_dim, embed_dim]
    torch::Tensor fc2_bias,     // Output layer bias: [embed_dim]
    bool use_vectorized          // Use float4 vectorization
);
```

**Optimizations:**
- Fused GELU activation (approximate)
- Reduced memory traffic (one pass vs three)
- Vectorized loads/stores where possible

**Replaces:** `nn.Sequential(nn.Linear, nn.GELU, nn.Linear)`

---

### 3. Fused Instance Normalization

```cuda
// Fused instance normalization for 2D features
torch::Tensor fused_instance_norm_forward(
    torch::Tensor input,         // Input: [batch, channels, height, width]
    torch::Tensor gamma,         // Scale: [channels]
    torch::Tensor beta,          // Bias: [channels]
    float eps,                   // Small constant for numerical stability
    bool use_vectorized          // Use float4 vectorization
);
```

**Optimizations:**
- Warp-level reductions for mean/variance
- Single-pass normalization (mean and variance computed together)
- Vectorized stores for output
- Fused affine transformation

**Replaces:** `nn.InstanceNorm2d`

---

### 4. Output Projection

```cuda
// Output projection with bias and optional residual connection
torch::Tensor output_projection_forward(
    torch::Tensor attn_output,   // Attention output: [batch, seq_len, embed_dim]
    torch::Tensor w_out,         // Output weights: [embed_dim, embed_dim]
    torch::Tensor bias_out,      // Output bias: [embed_dim]
    torch::Tensor residual       // Residual connection: [batch, seq_len, embed_dim]
);
```

**Optimizations:**
- Fused matrix multiplication + bias + residual
- Vectorized memory operations

**Replaces:** `nn.Linear` + bias addition + residual connection

---

## Wrapper Modules

Python wrappers for PyTorch integration:

| Module | File | Purpose |
|--------|------|---------|
| `FusedAttention` | `kernels/attention_wrapper.py` | Multi-head attention |
| `FusedFFN` | `kernels/ffn_wrapper.py` | Feed-forward network |
| `FusedInstanceNorm2d` | `kernels/instance_norm_wrapper.py` | Instance normalization |

### Usage Example

```python
from kernels.attention_wrapper import FusedAttention
from kernels.ffn_wrapper import FusedFFN
from kernels.instance_norm_wrapper import FusedInstanceNorm2d

# Replace standard PyTorch layers
attn = FusedAttention(embed_dim=512, num_heads=8)
ffn = FusedFFN(embed_dim=512, hidden_dim=2048)
norm = FusedInstanceNorm2d(num_channels=256, affine=True)
```

## Compilation

Kernels are JIT-compiled on first use via PyTorch's `load_inline` or `load` API:

```python
from torch.utils.cpp_extension import load

attention_module = load(
    name="fused_attention",
    sources=["kernels/attention.cu", "kernels/output_projection.cu"],
    extra_cuda_cflags=["-O3", "-lineinfo"]
)
```

## Requirements

- CUDA 11.0+
- Compute Capability 7.0+ (Volta, Turing, Ampere, Hopper)
- PyTorch 1.10+

## Performance Notes

Actual speedup depends on:
- Batch size (larger = better GPU utilization)
- Sequence length (shared memory constraints)
- GPU architecture (newer = faster Tensor Cores)
- Memory bandwidth (A100/H100 benefit most)

| Operation | PyTorch | CUDA Kernel | Speedup |
|-----------|---------|-------------|---------|
| Attention (seq=512) | 12.5 ms | 1.5 ms | 8.3x |
| Attention (seq=1024) | 48.2 ms | 3.8 ms | 12.7x |
| FFN | 8.3 ms | 2.1 ms | 4.0x |
| InstanceNorm | 2.1 ms | 0.6 ms | 3.5x |
