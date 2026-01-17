/*
StyleForge - Fused Multi-Head Attention Kernel (V2 - Register Accumulation)

V2 CHANGES:
- Use register-based accumulation to avoid large shared memory
- Compute scores and accumulate in a single pass (online softmax)
- Much simpler than batched approach

Expected speedup:
- seq_len=512: 2-3x over PyTorch baseline
- seq_len=1024: 3-5x over PyTorch baseline
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
// V2 KERNEL: Online Softmax - Single Pass
// -------------------------------------------------------------------------
template<int HEAD_DIM>
__global__ void attention_v2_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_qkv,
    const float* __restrict__ bias_qkv,
    float* __restrict__ head_outputs,
    int batch_size,
    int num_heads,
    int seq_len,
    int embed_dim,
    float scale
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_pos = blockIdx.z;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (batch_idx >= batch_size || head_idx >= num_heads || q_pos >= seq_len)
        return;

    const int head_dim = HEAD_DIM;

    // Shared memory for Q vector (all threads need to read it)
    __shared__ float s_q[HEAD_DIM];

    // ============================================================
    // Step 1: Compute Q once
    // ============================================================
    int64_t x_offset = ((int64_t)batch_idx * seq_len + q_pos) * embed_dim;
    int q_start_row = head_idx * head_dim;

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

    // Write Q to shared memory (only lane 0 of warp 0)
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            s_q[i] = q_local[i];
        }
    }
    __syncthreads();

    // Add Q bias
    if (tid == 0 && bias_qkv != nullptr) {
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            s_q[i] += bias_qkv[q_start_row + i];
        }
    }
    __syncthreads();

    // ============================================================
    // Step 2: Online softmax + V accumulation (single pass)
    // ============================================================
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output[HEAD_DIM] = {0};

    int k_start_row = embed_dim + head_idx * head_dim;
    int v_start_row = 2 * embed_dim + head_idx * head_dim;

    for (int k_pos = tid; k_pos < seq_len; k_pos += THREADS_PER_BLOCK) {
        int64_t x_k_offset = ((int64_t)batch_idx * seq_len + k_pos) * embed_dim;

        // Compute K
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

        // Compute Q·K score
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            score += s_q[i] * k_local[i];
        }
        score *= scale;

        // Online softmax update
        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_diff = expf(old_max - max_score);
        float exp_new = expf(score - max_score);

        sum_exp = sum_exp * exp_diff + exp_new;

        // Compute V and update output
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

        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            output[i] = output[i] * exp_diff + exp_new * v_local[i];
        }
    }

    // Reduce across threads
    float thread_max = max_score;
    max_score = warp_reduce_max(max_score);

    float scale_factor = expf(thread_max - max_score);
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        output[i] *= scale_factor;
    }
    sum_exp *= scale_factor;

    sum_exp = warp_reduce_sum(sum_exp);
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        output[i] = warp_reduce_sum(output[i]);
    }

    // Normalize and write output
    if (tid == 0) {
        sum_exp = fmaxf(sum_exp, 1e-8f);
        int64_t head_out_offset = ((int64_t)batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM + q_pos * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            head_outputs[head_out_offset + i] = output[i] / sum_exp;
        }
    }
}

// -------------------------------------------------------------------------
// Output Projection
// -------------------------------------------------------------------------
template<int HEAD_DIM>
__global__ void output_projection_v2_kernel(
    const float* __restrict__ head_outputs,
    const float* __restrict__ w_out,
    const float* __restrict__ bias_out,
    float* __restrict__ out,
    int num_heads,
    int batch_size,
    int seq_len,
    int embed_dim
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int out_dim = blockIdx.z;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || out_dim >= embed_dim)
        return;

    float sum = 0.0f;
    int head_idx = tid;

    if (head_idx < num_heads) {
        int64_t head_offset = ((int64_t)batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM + seq_idx * HEAD_DIM;
        int64_t w_out_offset = (int64_t)out_dim * embed_dim + head_idx * HEAD_DIM;

        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            sum += head_outputs[head_offset + i] * w_out[w_out_offset + i];
        }
    }

    sum = warp_reduce_sum(sum);
    sum = __shfl_sync(0xffffffff, sum, 0);

    if (tid == 0) {
        int64_t out_offset = ((int64_t)batch_idx * seq_len + seq_idx) * embed_dim + out_dim;
        float result = sum;
        if (bias_out != nullptr) {
            result += bias_out[out_dim];
        }
        out[out_offset] = result;
    }
}

// -------------------------------------------------------------------------
// Main Function
// -------------------------------------------------------------------------
torch::Tensor fused_attention_v2(
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

    auto out = torch::zeros_like(x);
    auto head_outputs = torch::zeros({batch_size, num_heads, seq_len, head_dim}, x.options());

    const float* bias_qkv_ptr = bias_qkv.has_value() ? bias_qkv.value().data_ptr<float>() : nullptr;
    const float* bias_out_ptr = bias_out.has_value() ? bias_out.value().data_ptr<float>() : nullptr;

    dim3 blocks1(batch_size, num_heads, seq_len);
    dim3 threads1(THREADS_PER_BLOCK);

    if (head_dim == 32) {
        attention_v2_kernel<32><<<blocks1, threads1>>>(
            x.data_ptr<float>(), w_qkv.data_ptr<float>(), bias_qkv_ptr,
            head_outputs.data_ptr<float>(), batch_size, num_heads,
            seq_len, embed_dim, scale);
    } else if (head_dim == 64) {
        attention_v2_kernel<64><<<blocks1, threads1>>>(
            x.data_ptr<float>(), w_qkv.data_ptr<float>(), bias_qkv_ptr,
            head_outputs.data_ptr<float>(), batch_size, num_heads,
            seq_len, embed_dim, scale);
    } else if (head_dim == 128) {
        attention_v2_kernel<128><<<blocks1, threads1>>>(
            x.data_ptr<float>(), w_qkv.data_ptr<float>(), bias_qkv_ptr,
            head_outputs.data_ptr<float>(), batch_size, num_heads,
            seq_len, embed_dim, scale);
    }

    // Output projection
    dim3 blocks2(batch_size, seq_len, embed_dim);
    dim3 threads2(32);

    if (head_dim == 32) {
        output_projection_v2_kernel<32><<<blocks2, threads2>>>(
            head_outputs.data_ptr<float>(), w_out.data_ptr<float>(), bias_out_ptr,
            out.data_ptr<float>(), num_heads, batch_size, seq_len, embed_dim);
    } else if (head_dim == 64) {
        output_projection_v2_kernel<64><<<blocks2, threads2>>>(
            head_outputs.data_ptr<float>(), w_out.data_ptr<float>(), bias_out_ptr,
            out.data_ptr<float>(), num_heads, batch_size, seq_len, embed_dim);
    } else if (head_dim == 128) {
        output_projection_v2_kernel<128><<<blocks2, threads2>>>(
            head_outputs.data_ptr<float>(), w_out.data_ptr<float>(), bias_out_ptr,
            out.data_ptr<float>(), num_heads, batch_size, seq_len, embed_dim);
    }

    return out;
}

// -------------------------------------------------------------------------
// Python Bindings
// -------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_attention_v2", &fused_attention_v2, "Fused attention V2 (online softmax)");
}
