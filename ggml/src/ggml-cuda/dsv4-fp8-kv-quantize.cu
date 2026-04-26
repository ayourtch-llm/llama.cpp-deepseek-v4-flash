#include "dsv4-fp8-kv-quantize.cuh"

static __device__ float dsv4_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * exp2f(float(exp - 7));
}

static __device__ float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    int best = 0;
    float best_diff = ax;
    for (int i = 1; i < 127; ++i) {
        const float val = dsv4_e4m3fn_value(i);
        const float diff = fabsf(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }

    return sign * dsv4_e4m3fn_value(best);
}

static __global__ void dsv4_fp8_kv_quantize_kernel(
        const char * __restrict__ src0,
        char * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const uint64_t nb00,
        const uint64_t nb01,
        const uint64_t nb02,
        const uint64_t nb03,
        const uint64_t nb0,
        const uint64_t nb1,
        const uint64_t nb2,
        const uint64_t nb3,
        const int32_t n_rot) {
    const int64_t n_nope = ne00 - n_rot;
    const int64_t n_rows = ne01 * ne02 * ne03;
    const int64_t row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }

    const int64_t i1 = row % ne01;
    const int64_t i2 = (row / ne01) % ne02;
    const int64_t i3 = row / (ne01 * ne02);

    const char * src_base = src0 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    char       * dst_base = dst  + i1 * nb1  + i2 * nb2  + i3 * nb3;

    extern __shared__ float scratch[];

    for (int64_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (threadIdx.x < 64) {
            v = *(const float *)(src_base + (off + threadIdx.x) * nb00);
            scratch[threadIdx.x] = fabsf(v);
        }
        __syncthreads();

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (threadIdx.x < 64) {
            const float q = dsv4_e4m3fn_dequant(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
            *(float *)(dst_base + (off + threadIdx.x) * nb0) = q;
        }
        __syncthreads();
    }

    for (int64_t i = n_nope + threadIdx.x; i < ne00; i += blockDim.x) {
        *(float *)(dst_base + i * nb0) = *(const float *)(src_base + i * nb00);
    }
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int32_t n_rot = ((int32_t *)dst->op_params)[0];
    const int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    cudaStream_t stream = ctx.stream();

    const int block_size = 64;
    const size_t shared_mem = 64 * sizeof(float);
    dsv4_fp8_kv_quantize_kernel<<<(int)n_rows, block_size, shared_mem, stream>>>(
        (const char *)src0->data,
        (char *)dst->data,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        n_rot);
}
