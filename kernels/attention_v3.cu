/*
StyleForge - Fused Multi-Head Attention Kernel (V3 - Register-Based)

V3 CHANGES:
- Register-based V accumulation (no shared memory for V)
- Warp reductions for softmax (online algorithm)
- Minimal shared memory: only Q vector
- Fixed nested loop issue

Key insight: Accumulate in registers, reduce across warps at the end.

Expected performance: Still slower than Flash Attention 2 (fundamental limitation),
but much better than V2. Educational value remains.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_BLOCK = 256;

// -------------------------------------------------------------------------
// Device Math Functions
// -------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// -------------------------------------------------------------------------
// V3 KERNEL: Register-Based Accumulation (Minimal Shared Memory)
// -------------------------------------------------------------------------
template<int HEAD_DIM>
__global__ void attention_v3_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_qkv,
    const float* __restrict__ bias_qkv,
    float* __restrict__ output,  // Direct output (no intermediate buffer)
    int batch_size,
    int num_heads,
    int seq_len,
    int embed_dim,
    float scale,
    const float* __restrict__ w_out,
    const float* __restrict__ bias_out
) {
    // Block: (batch, head, query_pos)
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_pos = blockIdx.z;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;

    if (batch_idx >= batch_size || head_idx >= num_heads || q_pos >= seq_len)
        return;

    const int head_dim = HEAD_DIM;

    // Shared memory: ONLY Q vector (tiny!)
    __shared__ float s_q[HEAD_DIM];

    int q_start_row = head_idx * head_dim;
    int k_start_row = embed_dim + head_idx * head_dim;
    int v_start_row = 2 * embed_dim + head_idx * head_dim;

    // ============================================================
    // Step 1: Compute Q once, store in shared memory
    // ============================================================
    int64_t x_offset = ((int64_t)batch_idx * seq_len + q_pos) * embed_dim;

    float q_local[HEAD_DIM] = {0};
    for (int k = tid; k < embed_dim; k += THREADS_PER_BLOCK) {
        float x_val = x[x_offset + k];
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            q_local[i] += x_val * w_qkv[(q_start_row + i) * embed_dim + k];
        }
    }

    // Warp reduction
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        q_local[i] = warp_reduce_sum(q_local[i]);
    }

    // Broadcast Q to all threads (lane 0 writes to shared)
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            s_q[i] = q_local[i];
        }
    }
    __syncthreads();

    // Add bias (thread 0)
    if (tid == 0 && bias_qkv != nullptr) {
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            s_q[i] += bias_qkv[q_start_row + i];
        }
    }
    __syncthreads();

    // ============================================================
    // Step 2: Online softmax + V accumulation (all in registers!)
    // ============================================================
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float v_accum[HEAD_DIM] = {0};

    // Each thread processes a subset of keys
    for (int k_pos = tid; k_pos < seq_len; k_pos += THREADS_PER_BLOCK) {
        int64_t x_k_offset = ((int64_t)batch_idx * seq_len + k_pos) * embed_dim;

        // --- Compute K ---
        float k_local[HEAD_DIM] = {0};
        for (int k = 0; k < embed_dim; k++) {
            float x_val = x[x_k_offset + k];
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                k_local[i] += x_val * w_qkv[(k_start_row + i) * embed_dim + k];
            }
        }
        if (bias_qkv != nullptr) {
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                k_local[i] += bias_qkv[k_start_row + i];
            }
        }

        // --- Compute Q·K score ---
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            score += s_q[i] * k_local[i];
        }
        score *= scale;

        // --- Online softmax update ---
        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_diff = expf(old_max - max_score);
        float exp_new = expf(score - max_score);

        sum_exp = sum_exp * exp_diff + exp_new;

        // --- Compute V ---
        float v_local[HEAD_DIM] = {0};
        for (int k = 0; k < embed_dim; k++) {
            float x_val = x[x_k_offset + k];
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                v_local[i] += x_val * w_qkv[(v_start_row + i) * embed_dim + k];
            }
        }
        if (bias_qkv != nullptr) {
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                v_local[i] += bias_qkv[v_start_row + i];
            }
        }

        // --- Accumulate weighted V (in registers!) ---
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            v_accum[i] = v_accum[i] * exp_diff + exp_new * v_local[i];
        }
    }

    // ============================================================
    // Step 3: Reduce across threads
    // ============================================================
    float thread_max = max_score;
    max_score = warp_reduce_max(max_score);

    float scale_factor = expf(thread_max - max_score);
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        v_accum[i] *= scale_factor;
    }
    sum_exp *= scale_factor;

    sum_exp = warp_reduce_sum(sum_exp);
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        v_accum[i] = warp_reduce_sum(v_accum[i]);
    }

    // ============================================================
    // Step 4: Write output (with output projection!)
    // ============================================================
    if (tid == 0) {
        sum_exp = fmaxf(sum_exp, 1e-8f);

        // Normalize
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            v_accum[i] /= sum_exp;
        }

        // Output projection: head_output @ w_out^T
        // This writes directly to final output, concatenated across heads
        int64_t out_offset = ((int64_t)batch_idx * seq_len + q_pos) * embed_dim + head_idx * head_dim;

        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            float sum = 0.0f;
            // Project to embed_dim output dimensions
            for (int j = 0; j < embed_dim; j++) {
                sum += v_accum[i] * w_out[j * embed_dim + head_idx * head_dim + i];
            }
            output[out_offset + i] = sum;
        }

        // Add bias (if this is the last head)
        if (bias_out != nullptr && head_idx == num_heads - 1) {
            int64_t row_offset = ((int64_t)batch_idx * seq_len + q_pos) * embed_dim;
            for (int d = 0; d < embed_dim; d++) {
                output[row_offset + d] += bias_out[d];
            }
        }
    }
}

// -------------------------------------------------------------------------
// Main Function
// -------------------------------------------------------------------------
torch::Tensor fused_attention_v3(
    torch::Tensor x,
    torch::Tensor w_qkv,
    torch::Tensor w_out,
    torch::optional<torch::Tensor> bias_qkv,
    torch::optional<torch::Tensor> bias_out,
    float scale,
    int64_t num_heads
) {
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int embed_dim = x.size(2);
    int head_dim = embed_dim / num_heads;

    auto options = x.options();

    // Output: [batch, seq_len, embed_dim]
    auto out = torch::zeros({batch_size, seq_len, embed_dim}, options);

    const float* bias_qkv_ptr = bias_qkv.has_value() ? bias_qkv.value().data_ptr<float>() : nullptr;
    const float* bias_out_ptr = bias_out.has_value() ? bias_out.value().data_ptr<float>() : nullptr;

    // Grid: one block per query position
    dim3 blocks(batch_size, num_heads, seq_len);
    dim3 threads(THREADS_PER_BLOCK);

    if (head_dim == 32) {
        attention_v3_kernel<32><<<blocks, threads>>>(
            x.data_ptr<float>(), w_qkv.data_ptr<float>(), bias_qkv_ptr,
            out.data_ptr<float>(), batch_size, num_heads,
            seq_len, embed_dim, scale,
            w_out.data_ptr<float>(), bias_out_ptr);
    } else if (head_dim == 64) {
        attention_v3_kernel<64><<<blocks, threads>>>(
            x.data_ptr<float>(), w_qkv.data_ptr<float>(), bias_qkv_ptr,
            out.data_ptr<float>(), batch_size, num_heads,
            seq_len, embed_dim, scale,
            w_out.data_ptr<float>(), bias_out_ptr);
    } else if (head_dim == 128) {
        attention_v3_kernel<128><<<blocks, threads>>>(
            x.data_ptr<float>(), w_qkv.data_ptr<float>(), bias_qkv_ptr,
            out.data_ptr<float>(), batch_size, num_heads,
            seq_len, embed_dim, scale,
            w_out.data_ptr<float>(), bias_out_ptr);
    }

    return out;
}

// -------------------------------------------------------------------------
// Python Bindings
// -------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_attention_v3", &fused_attention_v3, "Fused attention V3 (register-based)");
}
