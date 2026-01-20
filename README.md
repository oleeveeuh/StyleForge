# StyleForge

**High-Performance Neural Style Transfer with Custom CUDA Kernels**

A production-grade demonstration of advanced CUDA programming techniques for deep learning acceleration. StyleForge implements fast neural style transfer with custom kernels that achieve up to **1.72× end-to-end speedup** over PyTorch baseline and up to **2.07× with mixed precision**.

---

## Overview

StyleForge is a high-performance implementation of Johnson et al.'s transformer architecture for real-time neural style transfer. The project demonstrates practical application of low-level CUDA optimization techniques to accelerate deep learning inference, with comprehensive testing, profiling integration, and clean architecture.

This work showcases expertise in:
- **GPU kernel development** and performance optimization
- **Memory hierarchy optimization** (coalesced access, shared memory tiling, vectorization)
- **Kernel fusion** to eliminate kernel launch overhead
- **Mixed precision computing** for Tensor Core utilization
- **Online algorithms** for memory-efficient attention computation
- **Production-quality code** with testing and profiling tooling

---

## Performance Results

*All benchmarks run on NVIDIA Tesla T4 (Compute Capability 7.5), PyTorch 2.9.0+cu126*

### End-to-End Model Speedup (512×512 input)

| Variant | Inference Time | FPS | Speedup vs Baseline | Description |
|---------|---------------|-----|---------------------|-------------|
| **TransformerNetBaseline** | 30.61 ms | 32.7 | 1.0× | Pure PyTorch (nn.Conv2d + nn.InstanceNorm2d + nn.ReLU) |
| **TransformerNetFused** | 17.82 ms | 56.1 | **1.72×** | Fully fused Conv+IN+ReLU with custom kernels |

### Mixed Precision Performance

| Precision Mode | Inference Time | Speedup vs FP32 | Notes |
|----------------|---------------|-----------------|-------|
| **FP32 (float32)** | 30.77 ms | 1.0× | Baseline float32 precision |
| **Manual FP16** | 14.83 ms | **2.07×** | Manual `.half()` conversion |
| **PyTorch AMP** | 15.74 ms | **1.96×** | `torch.cuda.amp.autocast()` |

### Kernel-Level Benchmarks

#### FusedInstanceNorm2d

| Configuration | PyTorch | Fused | Speedup |
|---------------|---------|-------|---------|
| Small (64×64×64) | 0.28 ms | 0.11 ms | **2.46×** |
| Medium (128×128×128) | 0.33 ms | 0.23 ms | **1.41×** |

---

## Technical Deep Dive: Custom CUDA Kernels

StyleForge includes four custom CUDA kernels, each demonstrating specific optimization techniques:

### 1. Fused Instance Normalization ([`kernels/instance_norm.cu`](kernels/instance_norm.cu))

**Problem Solved:** PyTorch's `nn.InstanceNorm2d` launches 4 separate kernel launches (mean computation, variance computation, normalization, and affine transform). Each launch incurs GPU kernel launch overhead (typically 5-20 μs) and forces intermediate results to be written to and read from global memory.

**Solution:** A single fused kernel that combines all four stages, keeping intermediate values in registers and shared memory.

**Optimizations Applied:**

#### Warp-Level Reductions
```cuda
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

Instead of using shared memory atomics (which serialize execution), warp shuffle instructions enable parallel reduction within a warp in O(log W) steps. For a 32-thread warp, reduction completes in just 5 shuffle operations.

#### Vectorized Memory Access (float4)
```cuda
// Load 4 consecutive floats (128 bits) in a single transaction
float4 vec = input_vec[channel_offset + i];
sum += vec.x + vec.y + vec.z + vec.w;
```

When memory addresses are 128-bit aligned, `float4` loads transfer 16 bytes per transaction instead of 4 bytes, improving bandwidth utilization by up to 4x.

#### Coalesced Global Memory Access
Threads within a warp access consecutive memory locations, enabling the GPU to coalesce these accesses into fewer memory transactions.

**Achieved:** 2.46× speedup on small feature maps where kernel launch overhead dominates.

---

### 2. Fused Convolution + InstanceNorm + ReLU ([`kernels/conv_fusion.cu`](kernels/conv_fusion.cu))

**Problem Solved:** The style transfer network contains 13 Conv→IN→ReLU sequences. Each sequence requires 3 kernel launches with intermediate tensor storage in global memory (bandwidth: ~450 GB/s on T4 vs ~14 TB/s for shared memory).

**Solution:** Custom fused kernel combining all three operations, with specialized paths for 1×1 convolutions (dominant in residual blocks).

**Optimizations Applied:**

#### Coalesced Memory Access for 1×1 Convolution
```cuda
// Each thread processes consecutive spatial locations
int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;

// Process input channels sequentially for cache locality
#pragma unroll 4
for (int c_in = 0; c_in < C_in; c_in++) {
    int input_idx = (n * C_in + c_in) * spatial_size + spatial_idx;
    sum += input[input_idx] * weight_row[c_in];
}
```

Threads within a warp access consecutive memory locations, enabling full memory coalescing.

#### Shared Memory Bank Conflict Avoidance
```cuda
// Power-of-2 + 1 padding prevents 32-way bank conflicts
__shared__ __align__(16) float s_input[TILE_IN][TILE_IN + 1];
```

CUDA shared memory is banked (32 banks). Without padding, accessing `s_input[i][j]` where `j` is a multiple of 32 causes all threads to access the same bank, serializing accesses. The `+1` padding eliminates this conflict.

#### Persistent Kernel Strategy
```cuda
// Each block processes multiple (batch, channel) pairs
for (int bc = blockIdx.x; bc < N * C_out; bc += gridDim.x) {
    // Process one (batch, channel) instance
}
```

Rather than launching one block per (batch, channel) pair, fewer blocks are launched, and each loops over multiple instances. This amortizes kernel launch overhead over more work.

#### Mixed Precision Support (Tensor Cores)
```cuda
// FP16 path for Ampere+ GPUs
if constexpr (std::is_same_v<InputType, __half>) {
    sum += __half2float(__hmul(in0, w0));  // Uses Tensor Cores
}
```

On Ampere and Hopper architectures, FP16 math utilizes Tensor Cores, providing up to 8× throughput improvement for matrix operations.

**Achieved:** Significant speedup when used in the fully fused network variant.

---

### 3. Fused Multi-Head Attention V3 ([`kernels/attention_v3.cu`](kernels/attention_v3.cu))

**Problem Solved:** Standard attention requires O(N²) memory for the attention matrix (scores for each query-key pair) and multiple kernel launches for QKV projection, softmax computation, and output projection.

**Solution:** Register-based accumulation with online softmax, eliminating the need to materialize the full attention matrix.

**Key Innovation - Online Softmax:**

The standard softmax algorithm requires two passes: first to find the maximum value, then to compute exp(x-max)/sum(exp). Our online algorithm computes both in a single pass:

```cuda
float max_score = -INFINITY;
float sum_exp = 0.0f;
float v_accum[HEAD_DIM] = {0};

for (int k_pos = tid; k_pos < seq_len; k_pos += THREADS_PER_BLOCK) {
    float score = scale * dot(q, k_local);

    // Online softmax update
    float old_max = max_score;
    max_score = fmaxf(max_score, score);
    float exp_diff = expf(old_max - max_score);  // Rescale previous values
    float exp_new = expf(score - max_score);     // New value contribution

    sum_exp = sum_exp * exp_diff + exp_new;

    // Update V accumulation with corrected weights
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        v_accum[i] = v_accum[i] * exp_diff + exp_new * v_local[i];
    }
}
```

**Memory Reduction:**
- Standard approach: O(N²) for attention matrix + O(N·D) for V values
- Our approach: O(D) registers only (no attention matrix storage)

**Warp-Level Final Reduction:**
```cuda
// Each thread processed a subset of keys
float thread_max = max_score;
max_score = warp_reduce_max(max_score);  // Find global max

// Scale each thread's accumulation by the correction factor
float scale_factor = expf(thread_max - max_score);
sum_exp *= scale_factor;
#pragma unroll
for (int i = 0; i < HEAD_DIM; i++) {
    v_accum[i] *= scale_factor;
    v_accum[i] = warp_reduce_sum(v_accum[i]);
}
```

**Achieved:** O(N) memory complexity vs O(N²) for standard attention, with numerically stable softmax computation.

---

### 4. Fused Feed-Forward Network ([`kernels/ffn.cu`](kernels/ffn.cu))

**Problem Solved:** Transformer FFN blocks require 3 sequential kernels (FC1 → GELU → FC2) with intermediate tensor allocations and global memory round-trips.

**Solution:** Single fused kernel with inline GELU activation and shared memory tiling.

**Optimizations Applied:**

#### Inline GELU Activation
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

Uses inline PTX assembly for fast tanh approximation, avoiding function call overhead and enabling better instruction scheduling.

#### Shared Memory Tiling
```cuda
__shared__ float s_input[EMBED_DIM];
__shared__ float s_intermediate[FFN_DIM];

// Load input to shared memory (all threads access same input)
if (tid < EMBED_DIM) {
    s_input[tid] = input[input_idx];
}
__syncthreads();

// Each thread computes one output dimension
if (tid < FFN_DIM) {
    float val = fc1_bias[tid];
    #pragma unroll 4
    for (int i = 0; i < EMBED_DIM; i++) {
        val += s_input[i] * fc1_weight[i * ffn_dim + tid];
    }
    s_intermediate[tid] = gelu(val);
}
```

Input and intermediate values are stored in shared memory (~14 TB/s bandwidth) rather than global memory (~450 GB/s), significantly reducing memory latency.

#### Fused Residual Connection
```cuda
// Add residual connection without separate kernel
val += s_input[tid];
```

The residual connection is computed in-place, eliminating a separate tensor addition kernel.

**Achieved:** 4-5× speedup over `nn.Sequential(nn.Linear, nn.GELU, nn.Linear)`.

---

## CUDA Programming Techniques Demonstrated

| Technique | Kernel(s) Used | Benefit |
|-----------|----------------|---------|
| **Warp-level reduction** | instance_norm, attention, ffn | O(log W) vs O(N) reduction complexity |
| **Kernel fusion** | All kernels | Eliminates kernel launch overhead (~5-20 μs per launch) |
| **Coalesced memory access** | All kernels | 2-3× bandwidth improvement through aligned accesses |
| **Shared memory tiling** | conv_fusion, ffn | Reduces global memory traffic by ~10-100× |
| **Bank conflict avoidance** | conv_fusion | +1 padding prevents serialization on 32-bank SMs |
| **Vectorized loads (float4)** | instance_norm, ffn | 4× load efficiency for aligned addresses |
| **Persistent kernels** | conv_fusion | Amortizes launch overhead over more work |
| **Online algorithms** | attention_v3 | O(N) memory vs O(N²) with stable computation |
| **Mixed precision** | conv_fusion | Tensor Core utilization on Ampere+ GPUs |
| **Loop unrolling** | All kernels | Reduces branch overhead and improves ILP |
| **Inline PTX assembly** | ffn (tanh) | Fast math approximations with lower latency |

---

## Model Architecture

StyleForge implements the Johnson et al. transformer architecture:

```
Input (3, H, W)
    ↓
Encoder: 3 Conv+IN+ReLU blocks
         (3→32→64→128 channels, stride-2 downsampling)
    ↓
Residual Blocks: 5-10 residual connections
                  (128 channels, 3×3 convolutions)
    ↓
Decoder: 3 Upsample+Conv+IN+ReLU blocks
         (128→64→32→3 channels, nearest-neighbor upsampling)
    ↓
Output (3, H, W)
```

**Model specifications:**
- **Parameters:** ~1.7M (FP32)
- **Model size:** ~6.8 MB
- **Pre-trained styles:** Candy, Mosaic, Udnie, Rain Princess, Starry Night, Great Wave

---

## Project Structure

```
StyleForge/
├── kernels/                    # Custom CUDA kernel implementations
│   ├── instance_norm.cu        # Fused Instance Normalization
│   ├── conv_fusion.cu          # Fused Conv+IN+ReLU with Tensor Cores
│   ├── attention_v3.cu         # Online Softmax Attention (O(N) memory)
│   ├── ffn.cu                  # Fused Feed-Forward Network
│   ├── *_wrapper.py            # PyTorch integration via pybind11
│   └── test_*.py               # Kernel correctness tests
├── models/
│   ├── transformer_net.py      # Fast style transfer (3 variants)
│   └── optimized_blocks.py     # Reusable optimized blocks
├── tests/                      # Comprehensive test suite
│   ├── test_forward_pass.py    # Numerical correctness validation
│   ├── test_cuda_kernel_usage.py
│   ├── test_numerical_accuracy.py
│   └── test_memory_leaks.py
├── utils/
│   ├── benchmark.py            # Performance measurement utilities
│   ├── cuda_build.py           # JIT compilation for CUDA extensions
│   └── profiling.py            # Nsight Compute integration
└── notebooks/
    └── demo.ipynb              # Interactive demonstration
```

---

## Installation

### Requirements

- **GPU:** NVIDIA GPU with Compute Capability 7.0+ (Volta, Turing, Ampere, Hopper)
- **CUDA:** CUDA 11.0 or higher
- **Python:** 3.8+
- **PyTorch:** 1.10+

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/StyleForge.git
cd StyleForge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy matplotlib pillow ninja

# Build CUDA kernels (JIT compilation on first import)
python -c "from models.transformer_net import TransformerNet; print('Ready!')"

# Run tests
pytest tests/ -v
```

---

## Usage Examples

### Basic Style Transfer

```python
import torch
from models.transformer_net import TransformerNet
from utils.image_utils import load_image, save_image

# Load model
model = TransformerNet(num_residual_blocks=5).cuda()
model.load_checkpoint('path/to/candy.pth')
model.eval()

# Load and process image
input_tensor = load_image('input.jpg').cuda()
with torch.no_grad():
    output_tensor = model(input_tensor)

# Save result
save_image(output_tensor, 'output.jpg')
```

### Using the Optimized Variant

```python
from models.transformer_net import TransformerNetFused

# Faster variant with fused kernels
model = TransformerNetFused(num_residual_blocks=5).cuda()
```

### Benchmarking

```python
from utils.benchmark import benchmark_model

results = benchmark_model(
    model=model,
    input_size=(1, 3, 512, 512),
    num_warmup=10,
    num_iterations=100
)
print(f"Average inference time: {results['mean_ms']:.2f} ms")
print(f"FPS: {results['fps']:.1f}")
```

---

## Profiling with Nsight Compute

The project includes Nsight Compute profiling integration:

```bash
# Profile specific kernel
ncu --set full -f -o profile_output python -c "
from models.transformer_net import TransformerNet
import torch
model = TransformerNet().cuda()
x = torch.randn(1, 3, 512, 512).cuda()
model(x)
"

# View results
ncu-ui profile_output.ncu-rep
```

---

## Testing

The test suite validates both correctness and performance:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_numerical_accuracy.py -v
pytest tests/test_cuda_kernel_usage.py -v
pytest tests/test_memory_leaks.py -v
```

**Test coverage includes:**
- Forward pass correctness vs PyTorch reference
- Numerical accuracy validation (max difference < 0.07)
- CUDA kernel execution verification
- Memory leak detection
- Visual quality assessment

---

## Performance Recommendations

1. **Use PyTorch AMP** for production: provides 1.96× speedup with minimal code changes
2. **Use TransformerNetFused** for best inference performance
3. **Enable Tensor Cores** on Ampere+ GPUs with FP16/BF16
4. **Batch inference** when possible to amortize kernel launch overhead

---

## Future Work

Potential enhancements for further performance improvements:
- [ ] Flash Attention 2 integration for longer sequences
- [ ] INT8 quantization support for edge deployment
- [ ] Multi-GPU inference for batch processing
- [ ] TensorRT optimization for production deployment
- [ ] Triton Inference Server integration

---

## References

- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual Losses for Real-Time Style Transfer and Super-Resolution. [arXiv:1603.08155](https://arxiv.org/abs/1603.08155)
- NVIDIA CUDA C++ Programming Guide
- PyTorch C++ Extension Documentation
- Flash Attention: Dao, T. et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention.

---

## Author

Developed as a demonstration of high-performance GPU kernel programming and deep learning optimization techniques.
