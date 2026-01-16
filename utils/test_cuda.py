"""
StyleForge - Simple CUDA Kernel Tests

Tests basic CUDA functionality with minimal kernels.
Run this to verify CUDA JIT compilation is working before using complex kernels.
"""

import torch
from pathlib import Path


def test_simple_add_kernel():
    """
    Test a very simple vector addition kernel.
    This is the "Hello World" of CUDA programming.
    """
    cuda_source = """
// Simple vector addition: C = A + B
// Each thread computes one element
__global__ void vector_add(float* C, const float* A, const float* B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Python binding
torch::Tensor vector_add_forward(torch::Tensor A, torch::Tensor B) {
    auto C = torch::empty_like(A);
    int n = A.numel();

    // Launch kernel: 256 threads per block
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    vector_add<<<grid_size, block_size>>>(
        reinterpret_cast<float*>(C.data_ptr()),
        reinterpret_cast<const float*>(A.data_ptr()),
        reinterpret_cast<const float*>(B.data_ptr()),
        n
    );

    return C;
}
"""

    cpp_source = """
#include <torch/extension.h>

torch::Tensor vector_add_forward(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add_forward", &vector_add_forward, "Vector addition (CUDA)");
}
"""

    print("=" * 70)
    print("Testing Simple CUDA Vector Addition")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return False

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    try:
        print("\nCompiling simple vector add kernel...")
        from torch.utils.cpp_extension import load_inline

        module = load_inline(
            name="simple_vector_add",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=["-O3"],
            verbose=False
        )

        print("✓ Compilation successful!")

        # Test the kernel
        print("\nRunning kernel...")
        n = 1000000
        A = torch.randn(n, device='cuda')
        B = torch.randn(n, device='cuda')

        # Warmup
        for _ in range(10):
            C = module.vector_add_forward(A, B)
        torch.cuda.synchronize()

        # Timed run
        import time
        start = time.perf_counter()
        for _ in range(100):
            C = module.vector_add_forward(A, B)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / 100

        # Verify correctness
        expected = A + B
        max_diff = (C - expected).abs().max().item()

        print(f"\nResults:")
        print(f"  Input size: {n:,} elements")
        print(f"  Average time: {elapsed:.4f} ms")
        print(f"  Bandwidth: {(3 * n * 4 / elapsed / 1e6):.1f} MB/s")
        print(f"  Max error: {max_diff:.2e}")

        if max_diff < 1e-5:
            print("\n✅ SUCCESS! Simple CUDA kernel works correctly.")
            return True
        else:
            print(f"\n❌ FAILED: Output incorrect (max_diff={max_diff})")
            return False

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_reduction():
    """
    Test a simple reduction kernel (sum of array).
    This tests shared memory and synchronization.
    """
    cuda_source = """
// Simple reduction using atomic operations
__global__ void atomic_sum_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use atomicAdd to accumulate sum
    if (idx < n) {
        atomicAdd(&output[0], input[idx]);
    }
}

// Better reduction using shared memory
__global__ void shared_sum_kernel(float* output, const float* input, int n) {
    extern __shared__ float s_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load element into shared memory
    float value = (idx < n) ? input[idx] : 0.0f;
    s_data[tid] = value;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        output[blockIdx.x] = s_data[0];
    }
}

torch::Tensor sum_forward(torch::Tensor input) {
    int n = input.numel();
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // Partial sums
    auto partial_sums = torch::zeros(grid_size, input.options());

    // First pass: reduce within blocks
    shared_sum_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        reinterpret_cast<float*>(partial_sums.data_ptr()),
        reinterpret_cast<const float*>(input.data_ptr()),
        n
    );

    // Second pass: sum partial sums on CPU (simple for test)
    return torch::tensor({partial_sums.sum().item<float>()});
}
"""

    cpp_source = """
#include <torch/extension.h>

torch::Tensor sum_forward(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_forward", &sum_forward, "Sum reduction (CUDA)");
}
"""

    print("\n" + "=" * 70)
    print("Testing Simple CUDA Reduction")
    print("=" * 70)

    try:
        print("\nCompiling reduction kernel...")
        from torch.utils.cpp_extension import load_inline

        module = load_inline(
            name="simple_reduction",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=["-O3"],
            verbose=False
        )

        print("✓ Compilation successful!")

        # Test the kernel
        print("\nRunning kernel...")
        n = 100000
        input = torch.randn(n, device='cuda')

        # Warmup
        for _ in range(10):
            result = module.sum_forward(input)
        torch.cuda.synchronize()

        # Timed run
        import time
        start = time.perf_counter()
        for _ in range(100):
            result = module.sum_forward(input)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / 100

        # Verify correctness
        expected = input.sum().item()

        print(f"\nResults:")
        print(f"  Input size: {n:,} elements")
        print(f"  Average time: {elapsed:.4f} ms")
        print(f"  Sum: {result.item():.4f}")
        print(f"  Expected: {expected:.4f}")
        print(f"  Difference: {abs(result.item() - expected):.2e}")

        if abs(result.item() - expected) < 0.1:
            print("\n✅ SUCCESS! Reduction kernel works correctly.")
            return True
        else:
            print(f"\n❌ FAILED: Output incorrect")
            return False

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all simple CUDA tests."""
    print("\n" + "=" * 70)
    print("StyleForge CUDA JIT Tests")
    print("=" * 70)
    print("\nThese tests verify that CUDA JIT compilation is working.")
    print("If these pass, the issue is with the complex attention kernel.\n")

    results = {}

    results['vector_add'] = test_simple_add_kernel()
    results['reduction'] = test_simple_reduction()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✅ All tests passed! CUDA JIT is working.")
        print("The issue is likely with the attention kernel itself.")
    else:
        print("\n❌ Some tests failed. CUDA JIT may have issues.")

    return all_passed


def main():
    """Main entry point."""
    import sys
    run_all_tests()
    sys.exit(0 if run_all_tests() else 1)


if __name__ == "__main__":
    main()
