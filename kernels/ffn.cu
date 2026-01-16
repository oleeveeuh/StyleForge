/*
StyleForge - Fused Feed-Forward Network Kernel

Fuses: Linear → GELU → Linear → Bias → Residual

Key Optimizations:
- Single kernel launch for entire FFN block
- Shared memory for input and intermediate values
- Inline GELU activation
- Residual connection fused in
- Vectorized memory access

Performance Target: 4-5x speedup over PyTorch sequential implementation
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// ============================================
// CUDA Error Checking
// ============================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            std::abort(); \
        } \
    } while (0)

// ============================================
// Configuration
// ============================================
#define TILE_SIZE 16
#define WARP_SIZE 32

// ============================================
// GELU Activation (Inline)
// ============================================

__device__ __forceinline__ float gelu(float x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);

    // Fast tanh approximation using exp
    float tanh_val;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(tanh_val) : "f"(tanh_arg));

    return 0.5f * x * (1.0f + tanh_val);
}

// Alternative: Exact GELU using erf
__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678f));
}

// ============================================
// Vectorized GEMM Helper
// ============================================

template<int N>
__device__ __forceinline__ float dot_product(
    const float* __restrict__ a,
    const float* __restrict__ b,
    int offset_a,
    int offset_b,
    int stride_b
) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        sum += a[offset_a + i] * b[offset_b + i * stride_b];
    }
    return sum;
}

// ============================================
// Fused FFN Kernel V1
// ============================================

template<int EMBED_DIM, int FFN_DIM>
__global__ void fused_ffn_kernel_v1(
    const float* __restrict__ input,      // [B, S, E]
    const float* __restrict__ fc1_weight, // [E, F]
    const float* __restrict__ fc1_bias,   // [F]
    const float* __restrict__ fc2_weight, // [F, E]
    const float* __restrict__ fc2_bias,   // [E]
    float* __restrict__ output,           // [B, S, E]
    int batch_size,
    int seq_len,
    int embed_dim,
    int ffn_dim
) {
    // Grid: (seq_len, batch_size)
    int token_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= seq_len) return;

    // Shared memory for input and intermediate
    __shared__ float s_input[EMBED_DIM];
    __shared__ float s_intermediate[FFN_DIM];

    // Load input to shared memory
    if (tid < EMBED_DIM) {
        int input_idx = ((int64_t)batch_idx * seq_len + token_idx) * embed_dim + tid;
        s_input[tid] = input[input_idx];
    }
    __syncthreads();

    // ============================================
    // Stage 1: FC1 (Linear) + GELU Activation
    // ============================================

    if (tid < FFN_DIM) {
        float val = fc1_bias[tid];  // Start with bias

        // Matrix-vector multiply: input @ fc1_weight
        #pragma unroll 4
        for (int i = 0; i < EMBED_DIM; i++) {
            val += s_input[i] * fc1_weight[i * ffn_dim + tid];
        }

        // Apply GELU activation
        s_intermediate[tid] = gelu(val);
    }
    __syncthreads();

    // ============================================
    // Stage 2: FC2 (Linear) + Bias + Residual
    // ============================================

    if (tid < EMBED_DIM) {
        float val = fc2_bias[tid];  // Start with bias

        // Matrix-vector multiply: intermediate @ fc2_weight
        #pragma unroll 4
        for (int i = 0; i < FFN_DIM; i++) {
            val += s_intermediate[i] * fc2_weight[i * embed_dim + tid];
        }

        // Add residual connection
        val += s_input[tid];

        // Write output
        int out_idx = ((int64_t)batch_idx * seq_len + token_idx) * embed_dim + tid;
        output[out_idx] = val;
    }
}

// ============================================
// Fused FFN Kernel V2 (Optimized with float4)
// ============================================

template<int EMBED_DIM, int FFN_DIM>
__global__ void fused_ffn_kernel_v2(
    const float* __restrict__ input,
    const float* __restrict__ fc1_weight,
    const float* __restrict__ fc1_bias,
    const float* __restrict__ fc2_weight,
    const float* __restrict__ fc2_bias,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int embed_dim,
    int ffn_dim
) {
    // Vectorized memory loads using float4
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const float4* fc1_vec = reinterpret_cast<const float4*>(fc1_weight);
    float4* output_vec = reinterpret_cast<float4*>(output);

    int token_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= seq_len) return;

    // Shared memory (padded for float4 alignment)
    __shared__ float s_input[EMBED_DIM];
    __shared__ float s_intermediate[FFN_DIM];

    // Vectorized load of input
    int vec_size = embed_dim / 4;
    int input_vec_offset = ((int64_t)batch_idx * seq_len + token_idx) * vec_size;

    if (tid * 4 < EMBED_DIM) {
        float4 vec = input_vec[input_vec_offset + tid];
        s_input[tid * 4 + 0] = vec.x;
        s_input[tid * 4 + 1] = vec.y;
        s_input[tid * 4 + 2] = vec.z;
        s_input[tid * 4 + 3] = vec.w;
    }
    __syncthreads();

    // FC1 + GELU
    if (tid < FFN_DIM) {
        float val = fc1_bias[tid];
        #pragma unroll 4
        for (int i = 0; i < EMBED_DIM; i++) {
            val += s_input[i] * fc1_weight[i * ffn_dim + tid];
        }
        s_intermediate[tid] = gelu(val);
    }
    __syncthreads();

    // FC2 + Bias + Residual
    if (tid * 4 < EMBED_DIM) {
        float vals[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int out_dim = tid * 4 + j;
            if (out_dim < EMBED_DIM) {
                vals[j] = fc2_bias[out_dim];
                #pragma unroll 4
                for (int i = 0; i < FFN_DIM; i++) {
                    vals[j] += s_intermediate[i] * fc2_weight[i * embed_dim + out_dim];
                }
                vals[j] += s_input[out_dim];  // Residual
            }
        }

        // Vectorized store
        int out_vec_offset = ((int64_t)batch_idx * seq_len + token_idx) * vec_size + tid;
        if (tid * 4 < EMBED_DIM) {
            float4 vec;
            vec.x = vals[0];
            vec.y = vals[1];
            vec.z = vals[2];
            vec.w = vals[3];
            output_vec[out_vec_offset] = vec;
        }
    }
}

// ============================================
// Launcher Function
// ============================================

torch::Tensor fused_ffn_forward(
    torch::Tensor input,
    torch::Tensor fc1_weight,
    torch::Tensor fc1_bias,
    torch::Tensor fc2_weight,
    torch::Tensor fc2_bias,
    bool use_vectorized = true
) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int embed_dim = input.size(2);
    const int ffn_dim = fc1_bias.size(0);

    auto output = torch::zeros_like(input);

    dim3 block(512);  // Threads per block
    dim3 grid(seq_len, batch_size);

    int smem_size = sizeof(float) * (embed_dim + ffn_dim);

    // Launch appropriate kernel based on dimensions
    // Since template parameters must be compile-time constants,
    // we use a series of if-else checks

    if (embed_dim == 128 && ffn_dim == 512) {
        if (use_vectorized) {
            fused_ffn_kernel_v2<128, 512><<<grid, block, smem_size>>>(
                input.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(),
                fc2_bias.data_ptr<float>(), output.data_ptr<float>(),
                batch_size, seq_len, embed_dim, ffn_dim);
        } else {
            fused_ffn_kernel_v1<128, 512><<<grid, block, smem_size>>>(
                input.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(),
                fc2_bias.data_ptr<float>(), output.data_ptr<float>(),
                batch_size, seq_len, embed_dim, ffn_dim);
        }
    } else if (embed_dim == 256 && ffn_dim == 1024) {
        if (use_vectorized) {
            fused_ffn_kernel_v2<256, 1024><<<grid, block, smem_size>>>(
                input.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(),
                fc2_bias.data_ptr<float>(), output.data_ptr<float>(),
                batch_size, seq_len, embed_dim, ffn_dim);
        } else {
            fused_ffn_kernel_v1<256, 1024><<<grid, block, smem_size>>>(
                input.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(),
                fc2_bias.data_ptr<float>(), output.data_ptr<float>(),
                batch_size, seq_len, embed_dim, ffn_dim);
        }
    } else if (embed_dim == 512 && ffn_dim == 2048) {
        if (use_vectorized) {
            fused_ffn_kernel_v2<512, 2048><<<grid, block, smem_size>>>(
                input.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(),
                fc2_bias.data_ptr<float>(), output.data_ptr<float>(),
                batch_size, seq_len, embed_dim, ffn_dim);
        } else {
            fused_ffn_kernel_v1<512, 2048><<<grid, block, smem_size>>>(
                input.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(),
                fc2_bias.data_ptr<float>(), output.data_ptr<float>(),
                batch_size, seq_len, embed_dim, ffn_dim);
        }
    } else if (embed_dim == 768 && ffn_dim == 3072) {
        if (use_vectorized) {
            fused_ffn_kernel_v2<768, 3072><<<grid, block, smem_size>>>(
                input.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(),
                fc2_bias.data_ptr<float>(), output.data_ptr<float>(),
                batch_size, seq_len, embed_dim, ffn_dim);
        } else {
            fused_ffn_kernel_v1<768, 3072><<<grid, block, smem_size>>>(
                input.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(),
                fc2_bias.data_ptr<float>(), output.data_ptr<float>(),
                batch_size, seq_len, embed_dim, ffn_dim);
        }
    } else if (embed_dim == 1024 && ffn_dim == 4096) {
        if (use_vectorized) {
            fused_ffn_kernel_v2<1024, 4096><<<grid, block, smem_size>>>(
                input.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(),
                fc2_bias.data_ptr<float>(), output.data_ptr<float>(),
                batch_size, seq_len, embed_dim, ffn_dim);
        } else {
            fused_ffn_kernel_v1<1024, 4096><<<grid, block, smem_size>>>(
                input.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(),
                fc2_bias.data_ptr<float>(), output.data_ptr<float>(),
                batch_size, seq_len, embed_dim, ffn_dim);
        }
    } else {
        // Generic fallback - use PyTorch for unsupported dimensions
        // For now, return the output as-is (no-op)
        // In production, we'd want to either:
        // 1. Add more template specializations, or
        // 2. Fall back to a non-templated kernel
        TORCH_CHECK(false,
            "Unsupported FFN dimensions: embed_dim=", embed_dim,
            ", ffn_dim=", ffn_dim, ". Supported: (128,512), (256,1024), (512,2048), (768,3072), (1024,4096)");
    }

    CUDA_CHECK(cudaGetLastError());

    return output;
}

// ============================================
// Pybind11 Module
// ============================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_ffn_forward, "Fused FFN (CUDA)");
}
