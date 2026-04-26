#include "dsv4-hc-split-sinkhorn.cuh"

#define HC_MAX 16

static __global__ void dsv4_hc_split_sinkhorn_kernel(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float * __restrict__ dst,
        const int n_hc,
        const int sinkhorn_iters,
        const float eps,
        const int64_t mix_hc,
        const int64_t n_rows,
        const uint64_t nb01,
        const uint64_t nb1) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }

    const float * mix = (const float *)((const char *)mixes + row * nb01);
    float * out       = (float *)((char *)dst + row * nb1);

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < n_hc; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    for (int i = 0; i < n_hc; ++i) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[HC_MAX * HC_MAX];

    for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            c[idx] = c[idx] * inv_sum + eps;
        }
    }

    for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            sum += c[src_hc + dst_hc * n_hc];
        }
        const float inv_denom = 1.0f / (sum + eps);
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            c[src_hc + dst_hc * n_hc] *= inv_denom;
        }
    }

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < n_hc * n_hc; ++i) {
        out[2 * n_hc + i] = c[i];
    }
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);

    const int n_hc           = ((int32_t *)dst->op_params)[0];
    const int sinkhorn_iters = ((int32_t *)dst->op_params)[1];
    float eps;
    memcpy(&eps, (int32_t *)dst->op_params + 2, sizeof(float));

    const int64_t mix_hc = mixes->ne[0];
    const int64_t n_rows = ggml_nrows(mixes);

    cudaStream_t stream = ctx.stream();

    const int num_blocks = (int)((n_rows + CUDA_DSV4_HC_SINKHORN_BLOCK_SIZE - 1) / CUDA_DSV4_HC_SINKHORN_BLOCK_SIZE);
    dsv4_hc_split_sinkhorn_kernel<<<num_blocks, CUDA_DSV4_HC_SINKHORN_BLOCK_SIZE, 0, stream>>>(
        (const float *)mixes->data,
        (const float *)scale->data,
        (const float *)base->data,
        (float *)dst->data,
        n_hc, sinkhorn_iters, eps, mix_hc, n_rows,
        mixes->nb[1], dst->nb[1]);
}
