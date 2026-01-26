/*
StyleForge - Fused Multi-Head Attention Kernel (V5 - Fixed Grid Config)

V5 FIXES:
- V4 had catastrophically bad grid: (batch, heads, seq_len) blocks
- For seq_len=512, heads=16: launches 8,192 blocks (massive overhead!)
- V5: One block per HEAD, processes ALL query positions
- Grid: (batch_size, num_heads) blocks instead of (batch_size, num_heads, seq_len)
- Each block loops over query positions, keeping K/V in shared memory for reuse
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
// V5 KERNEL: One block per head, process all query positions
// -------------------------------------------------------------------------
template<int HEAD_DIM>
__global__ void attention_v5_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Block: (batch, head) - ONE block per head, NOT per query position!
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (batch_idx >= batch_size || head_idx >= num_heads)
        return;

    // Base pointers for this batch and head
    int64_t head_offset = ((int64_t)batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const float* Q_head = Q + head_offset;
    const float* K_head = K + head_offset;
    const float* V_head = V + head_offset;

    // Each warp handles a subset of query positions
    // We have 8 warps, distribute seq_len across them
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;  // 8

    // Each warp processes multiple query positions
    for (int q_pos_base = warp_id; q_pos_base < seq_len; q_pos_base += WARPS_PER_BLOCK) {
        int q_pos = q_pos_base;

        // Load Q vector for this query position into registers
        float q_local[HEAD_DIM];
        #pragma unroll
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            q_local[i] = Q_head[q_pos * HEAD_DIM + i];
        }

        // Sync warp so all threads have their Q values
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            q_local[i] = __shfl_sync(0xffffffff, q_local[i], i / (HEAD_DIM / WARP_SIZE));
        }

        // Online softmax + V accumulation
        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float v_accum[HEAD_DIM] = {0};

        // Each thread processes a subset of key positions
        for (int k_pos = tid; k_pos < seq_len; k_pos += THREADS_PER_BLOCK) {
            // Load K vector
            float k_local[HEAD_DIM];
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                k_local[i] = K_head[k_pos * HEAD_DIM + i];
            }

            // Compute Q·K score
            float score = 0.0f;
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                score += q_local[i] * k_local[i];
            }
            score *= scale;

            // Online softmax update
            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            float exp_new = expf(score - max_score);

            sum_exp = sum_exp * exp_diff + exp_new;

            // Load V vector
            float v_local[HEAD_DIM];
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                v_local[i] = V_head[k_pos * HEAD_DIM + i];
            }

            // Accumulate weighted V
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                v_accum[i] = v_accum[i] * exp_diff + exp_new * v_local[i];
            }
        }

        // ============================================================
        // Reduce within warp
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
        // Reduce across warps using shared memory
        // ============================================================
        __shared__ float s_warp_max[WARPS_PER_BLOCK];
        __shared__ float s_warp_sum[WARPS_PER_BLOCK];
        __shared__ float s_warp_v[WARPS_PER_BLOCK][HEAD_DIM];

        if (lane_id == 0) {
            s_warp_max[warp_id] = max_score;
            s_warp_sum[warp_id] = sum_exp;
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                s_warp_v[warp_id][i] = v_accum[i];
            }
        }
        __syncthreads();

        // First warp combines results from all warps
        if (warp_id == 0) {
            float global_max = s_warp_max[0];
            float global_sum = s_warp_sum[0];
            float global_v[HEAD_DIM];

            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                global_v[i] = s_warp_v[0][i];
            }

            for (int w = 1; w < WARPS_PER_BLOCK; w++) {
                float warp_max = s_warp_max[w];
                float warp_sum = s_warp_sum[w];

                float old_global_max = global_max;
                global_max = fmaxf(global_max, warp_max);

                float rescale_global = expf(old_global_max - global_max);
                float rescale_warp = expf(warp_max - global_max);

                global_sum = global_sum * rescale_global + warp_sum * rescale_warp;

                #pragma unroll
                for (int i = 0; i < HEAD_DIM; i++) {
                    global_v[i] = global_v[i] * rescale_global + s_warp_v[w][i] * rescale_warp;
                }
            }

            // Thread 0 writes output
            if (lane_id == 0) {
                global_sum = fmaxf(global_sum, 1e-8f);
                int64_t out_offset = head_offset + q_pos * HEAD_DIM;

                #pragma unroll
                for (int i = 0; i < HEAD_DIM; i++) {
                    output[out_offset + i] = global_v[i] / global_sum;
                }
            }
        }
        __syncthreads();  // Sync before processing next query position
    }
}

// -------------------------------------------------------------------------
// Main Function - Takes pre-computed Q, K, V
// -------------------------------------------------------------------------
torch::Tensor fused_attention_v5(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    int batch_size = Q.size(0);
    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);

    auto options = Q.options();

    // Output: [batch, num_heads, seq_len, head_dim]
    auto out = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);

    // CRITICAL FIX: Grid is (batch, heads) NOT (batch, heads, seq_len)!
    // For seq_len=512, heads=16:
    //   V4: (1, 16, 512) = 8,192 blocks (catastrophic!)
    //   V5: (1, 16) = 16 blocks (efficient!)
    dim3 blocks(batch_size, num_heads);
    dim3 threads(THREADS_PER_BLOCK);

    if (head_dim == 32) {
        attention_v5_kernel<32><<<blocks, threads>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            out.data_ptr<float>(), batch_size, num_heads, seq_len, head_dim, scale
        );
    } else if (head_dim == 64) {
        attention_v5_kernel<64><<<blocks, threads>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            out.data_ptr<float>(), batch_size, num_heads, seq_len, head_dim, scale
        );
    } else if (head_dim == 128) {
        attention_v5_kernel<128><<<blocks, threads>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            out.data_ptr<float>(), batch_size, num_heads, seq_len, head_dim, scale
        );
    }

    return out;
}

// -------------------------------------------------------------------------
// Python Bindings
// -------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_attention_v5", &fused_attention_v5, "Fused attention V5 (fixed grid config)");
}
