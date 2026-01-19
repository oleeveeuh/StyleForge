/*
StyleForge - Test CUDA Kernels

Simple kernels for verifying CUDA compilation and testing
optimization techniques.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// -------------------------------------------------------------------------
// Error checking macro
// -------------------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while(0)

// -------------------------------------------------------------------------
// Kernel 1: Simple element-wise multiplication
// -------------------------------------------------------------------------
__global__ void multiply_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

torch::Tensor multiply_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Input a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Input b must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input b must be float32");

    auto c = torch::zeros_like(a);

    int size = a.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    multiply_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        size
    );
    CUDA_CHECK(cudaGetLastError());

    return c;
}

// -------------------------------------------------------------------------
// Kernel 2: Vectorized element-wise multiplication (float4)
// -------------------------------------------------------------------------
__global__ void multiply_vectorized_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        // Vectorized load using float4 (4 floats = 128 bits)
        float4 a4 = reinterpret_cast<const float4*>(a)[idx / 4];
        float4 b4 = reinterpret_cast<const float4*>(b)[idx / 4];

        // Element-wise multiply
        float4 c4;
        c4.x = a4.x * b4.x;
        c4.y = a4.y * b4.y;
        c4.z = a4.z * b4.z;
        c4.w = a4.w * b4.w;

        // Vectorized store
        reinterpret_cast<float4*>(c)[idx / 4] = c4;
    }
}

torch::Tensor multiply_vectorized_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Input a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Input b must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input b must be float32");

    auto c = torch::zeros_like(a);

    int size = a.numel();
    const int threads = 256;
    const int blocks = ((size / 4) + threads - 1) / threads;

    multiply_vectorized_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        size
    );
    CUDA_CHECK(cudaGetLastError());

    return c;
}

// -------------------------------------------------------------------------
// Kernel 3: Shared memory reduction (sum)
// -------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Shared memory for block-level reduction
    __shared__ float sdata[BLOCK_SIZE];

    // Load element (0 if out of bounds)
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduce in shared memory
    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

torch::Tensor sum_cuda(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    int size = input.numel();
    const int BLOCK_SIZE = 256;
    const int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate intermediate output
    auto partial_sums = torch::zeros({blocks}, torch::dtype(torch::kFloat32).device(input.device()));

    // First level reduction
    sum_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        size
    );
    CUDA_CHECK(cudaGetLastError());

    // Final reduction on CPU (or could do another kernel pass)
    auto result = partial_sums.sum();

    return result;
}

// -------------------------------------------------------------------------
// Kernel 4: Fused multiply-add (a * b + c)
// -------------------------------------------------------------------------
__global__ void multiply_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ d,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d[idx] = a[idx] * b[idx] + c[idx];  // FMA: one instruction
    }
}

torch::Tensor multiply_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.device().is_cuda(), "Input a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Input b must be on CUDA");
    TORCH_CHECK(c.device().is_cuda(), "Input c must be on CUDA");

    auto d = torch::zeros_like(a);

    int size = a.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    multiply_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        d.data_ptr<float>(),
        size
    );
    CUDA_CHECK(cudaGetLastError());

    return d;
}

// -------------------------------------------------------------------------
// Pybind11 module definition
// -------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multiply", &multiply_cuda, "Element-wise multiply (CUDA)");
    m.def("multiply_vectorized", &multiply_vectorized_cuda, "Element-wise multiply with float4 vectorization");
    m.def("sum", &sum_cuda, "Sum reduction using shared memory");
    m.def("multiply_add", &multiply_add_cuda, "Fused multiply-add (a * b + c)");
}
