#include "dsv4-rope-tail.cuh"
#include "ggml.h"

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static __device__ void dsv4_rope_yarn(
        const float theta_extrap, const float freq_scale, const float corr_dim0, const float corr_dim1,
        const int64_t i0, const float ext_factor, float mscale,
        float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dim0, corr_dim1, i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

template <typename T>
static __global__ void dsv4_rope_tail_kernel(
        const T * __restrict__ src0,
        T * __restrict__ dst,
        const int32_t * __restrict__ pos,
        const float * __restrict__ freq_factors,
        const int64_t ne00,
        const uint64_t nb00,
        const uint64_t nb01,
        const uint64_t nb02,
        const uint64_t nb03,
        const uint64_t nb0,
        const uint64_t nb1,
        const uint64_t nb2,
        const uint64_t nb3,
        const int n_dims,
        const int mode,
        const int inverse,
        const float freq_base,
        const float freq_scale,
        const float ext_factor,
        const float attn_factor,
        const float corr_dim0,
        const float corr_dim1,
        const bool has_ff) {

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y;
    const int64_t i3 = blockIdx.z;

    const int64_t n_nope = ne00 - n_dims;

    const float theta_base = (float)pos[i2];
    const float inv_ndims = -1.0f / n_dims;
    const bool is_neox = (mode == 2);

    const char * src_row = (const char *)src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    char       * dst_row = (char *)dst        + i3 * nb3  + i2 * nb2  + i1 * nb1;

    for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
        if (i0 < n_nope) {
            *(T *)(dst_row + i0 * nb0) = *(const T *)(src_row + i0 * nb00);
            continue;
        }

        const int r = (int)(i0 - n_nope);

        if (is_neox) {
            const int n_half = n_dims / 2;
            if (r >= n_half) {
                continue;
            }

            const int ic = r;
            const int rel_i0 = 2 * ic;
            const float theta = theta_base * powf(freq_base, inv_ndims * rel_i0);
            const float freq_factor = has_ff ? freq_factors[ic] : 1.0f;

            float cos_theta, sin_theta;
            dsv4_rope_yarn(theta / freq_factor, freq_scale, corr_dim0, corr_dim1, rel_i0, ext_factor, attn_factor, cos_theta, sin_theta);
            if (inverse) {
                sin_theta = -sin_theta;
            }

            const int64_t j0 = n_nope + ic;
            const int64_t j1 = n_nope + ic + n_half;
            const float x0 = (float)(*(const T *)(src_row + j0 * nb00));
            const float x1 = (float)(*(const T *)(src_row + j1 * nb00));

            *(T *)(dst_row + j0 * nb0) = (T)(x0 * cos_theta - x1 * sin_theta);
            *(T *)(dst_row + j1 * nb0) = (T)(x0 * sin_theta + x1 * cos_theta);
        } else {
            if ((r & 1) != 0) {
                continue;
            }

            const int ic = r / 2;
            const float theta = theta_base * powf(freq_base, inv_ndims * r);
            const float freq_factor = has_ff ? freq_factors[ic] : 1.0f;

            float cos_theta, sin_theta;
            dsv4_rope_yarn(theta / freq_factor, freq_scale, corr_dim0, corr_dim1, r, ext_factor, attn_factor, cos_theta, sin_theta);
            if (inverse) {
                sin_theta = -sin_theta;
            }

            const int64_t j0 = n_nope + r;
            const int64_t j1 = j0 + 1;
            const float x0 = (float)(*(const T *)(src_row + j0 * nb00));
            const float x1 = (float)(*(const T *)(src_row + j1 * nb00));

            *(T *)(dst_row + j0 * nb0) = (T)(x0 * cos_theta - x1 * sin_theta);
            *(T *)(dst_row + j1 * nb0) = (T)(x0 * sin_theta + x1 * cos_theta);
        }
    }
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == src0->type);

    const int n_dims     = ((int32_t *)dst->op_params)[0];
    const int mode       = ((int32_t *)dst->op_params)[1];
    const int n_ctx_orig = ((int32_t *)dst->op_params)[2];
    const int inverse    = ((int32_t *)dst->op_params)[3];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *)dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (int32_t *)dst->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (int32_t *)dst->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (int32_t *)dst->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (int32_t *)dst->op_params + 9, sizeof(float));

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const float * freq_factors = src2 ? (const float *)src2->data : nullptr;
    const bool has_ff = (src2 != nullptr);

    cudaStream_t stream = ctx.stream();

    const int block_size = 256;
    dim3 grid(src0->ne[1], src0->ne[2], src0->ne[3]);

    if (src0->type == GGML_TYPE_F32) {
        dsv4_rope_tail_kernel<float><<<grid, block_size, 0, stream>>>(
            (const float *)src0->data,
            (float *)dst->data,
            (const int32_t *)src1->data,
            freq_factors,
            src0->ne[0],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
            n_dims, mode, inverse,
            freq_base, freq_scale, ext_factor, attn_factor,
            corr_dims[0], corr_dims[1], has_ff);
    } else {
        dsv4_rope_tail_kernel<half><<<grid, block_size, 0, stream>>>(
            (const half *)src0->data,
            (half *)dst->data,
            (const int32_t *)src1->data,
            freq_factors,
            src0->ne[0],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
            n_dims, mode, inverse,
            freq_base, freq_scale, ext_factor, attn_factor,
            corr_dims[0], corr_dims[1], has_ff);
    }
}
