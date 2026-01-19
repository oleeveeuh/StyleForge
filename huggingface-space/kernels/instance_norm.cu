/*
StyleForge - Fused Instance Normalization Kernel

Fuses: Mean → Variance → Normalize → Affine Transform

Key Optimizations:
- Single kernel launch for entire InstanceNorm operation
- Warp-level reductions for mean/variance computation
- Fused affine transform (gamma * normalized + beta)
- Efficient shared memory usage

Performance Target: 3-5x speedup over PyTorch nn.InstanceNorm2d
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
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024

// ============================================
// Warp-Level Primitives
// ============================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ============================================
// Fused Instance Norm Kernel
// ============================================

template<int BLOCK_SIZE>
__global__ void fused_instance_norm_kernel(
    const float* __restrict__ input,   // [B, C, H, W]
    const float* __restrict__ gamma,   // [C]
    const float* __restrict__ beta,    // [C]
    float* __restrict__ output,        // [B, C, H, W]
    int batch_size,
    int channels,
    int height,
    int width,
    float eps
) {
    // One block per (batch, channel) instance
    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.x;
    int tid = threadIdx.x;
    int spatial_size = height * width;

    // Shared memory for reductions
    __shared__ float s_warp_sums[32];  // Up to 32 warps
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    // Input offset for this (batch, channel)
    int64_t channel_offset = ((int64_t)batch_idx * channels + channel_idx) * spatial_size;

    // ============================================
    // Stage 1: Compute Mean
    // ============================================

    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += BLOCK_SIZE) {
        sum += input[channel_offset + i];
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Store warp sum in shared memory
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        s_warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid == 0) {
        float total = 0.0f;
        int num_warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total += s_warp_sums[i];
        }
        s_mean = total / spatial_size;
    }
    __syncthreads();

    float mean = s_mean;

    // ============================================
    // Stage 2: Compute Variance
    // ============================================

    float var_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += BLOCK_SIZE) {
        float diff = input[channel_offset + i] - mean;
        var_sum += diff * diff;
    }

    // Warp-level reduction
    var_sum = warp_reduce_sum(var_sum);

    if (lane_id == 0) {
        s_warp_sums[warp_id] = var_sum;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid == 0) {
        float total = 0.0f;
        int num_warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total += s_warp_sums[i];
        }
        float variance = total / spatial_size;
        s_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    float inv_std = s_inv_std;

    // ============================================
    // Stage 3: Normalize & Affine Transform (Fused)
    // ============================================

    float gamma_val = gamma[channel_idx];
    float beta_val = beta[channel_idx];

    for (int i = tid; i < spatial_size; i += BLOCK_SIZE) {
        int idx = channel_offset + i;

        // Normalize: (x - mean) / std
        float normalized = (input[idx] - mean) * inv_std;

        // Affine transform: gamma * x + beta
        output[idx] = gamma_val * normalized + beta_val;
    }
}

// ============================================
// Vectorized Instance Norm (float4)
// ============================================

template<int BLOCK_SIZE>
__global__ void fused_instance_norm_kernel_vec4(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    float eps
) {
    // Vectorized loads using float4 (4 pixels at once)
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);

    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.x;
    int tid = threadIdx.x;
    int spatial_size = height * width;
    int vec_size = spatial_size / 4;

    __shared__ float s_warp_sums[32];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    int64_t channel_offset = ((int64_t)batch_idx * channels + channel_idx) * vec_size;

    // Compute mean using vectorized loads
    float sum = 0.0f;
    for (int i = tid; i < vec_size; i += BLOCK_SIZE) {
        float4 vec = input_vec[channel_offset + i];
        sum += vec.x + vec.y + vec.z + vec.w;
    }

    sum = warp_reduce_sum(sum);

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        s_warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int num_warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total += s_warp_sums[i];
        }
        s_mean = total / spatial_size;
    }
    __syncthreads();

    float mean = s_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < vec_size; i += BLOCK_SIZE) {
        float4 vec = input_vec[channel_offset + i];
        float4 diff;
        diff.x = vec.x - mean;
        diff.y = vec.y - mean;
        diff.z = vec.z - mean;
        diff.w = vec.w - mean;
        var_sum += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
    }

    var_sum = warp_reduce_sum(var_sum);

    if (lane_id == 0) {
        s_warp_sums[warp_id] = var_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int num_warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total += s_warp_sums[i];
        }
        float variance = total / spatial_size;
        s_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    float inv_std = s_inv_std;
    float gamma_val = gamma[channel_idx];
    float beta_val = beta[channel_idx];

    // Normalize and apply affine transform
    for (int i = tid; i < vec_size; i += BLOCK_SIZE) {
        float4 vec = input_vec[channel_offset + i];
        float4 result;
        result.x = gamma_val * (vec.x - mean) * inv_std + beta_val;
        result.y = gamma_val * (vec.y - mean) * inv_std + beta_val;
        result.z = gamma_val * (vec.z - mean) * inv_std + beta_val;
        result.w = gamma_val * (vec.w - mean) * inv_std + beta_val;
        output_vec[channel_offset + i] = result;
    }
}

// ============================================
// Launcher Function
// ============================================

torch::Tensor fused_instance_norm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps,
    bool use_vectorized
) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (B, C, H, W)");

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int spatial_size = height * width;

    auto output = torch::zeros_like(input);

    dim3 block(256);
    dim3 grid(channels, batch_size);

    // Use vectorized kernel if spatial size is multiple of 4
    bool use_vec4 = use_vectorized && (spatial_size % 4 == 0);

    if (use_vec4) {
        fused_instance_norm_kernel_vec4<256><<<grid, block>>>(
            input.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            height,
            width,
            eps
        );
    } else {
        fused_instance_norm_kernel<256><<<grid, block>>>(
            input.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            height,
            width,
            eps
        );
    }

    CUDA_CHECK(cudaGetLastError());

    return output;
}

// ============================================
// Pybind11 Module
// ============================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_instance_norm_forward, "Fused InstanceNorm (CUDA)");
}
