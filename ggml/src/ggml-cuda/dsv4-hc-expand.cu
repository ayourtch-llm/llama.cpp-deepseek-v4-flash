#include "dsv4-hc-expand.cuh"

static __global__ void dsv4_hc_expand_kernel(
        const char * __restrict__ block_out,
        const char * __restrict__ residual,
        const char * __restrict__ post,
        const char * __restrict__ comb,
        char * __restrict__ dst,
        const int64_t n_embd,
        const int64_t n_hc,
        const int64_t n_tokens,
        const uint64_t nb_block0,
        const uint64_t nb_block1,
        const uint64_t nb_res0,
        const uint64_t nb_res1,
        const uint64_t nb_res2,
        const uint64_t nb_post0,
        const uint64_t nb_post1,
        const uint64_t nb_comb0,
        const uint64_t nb_comb1,
        const uint64_t nb_comb2,
        const uint64_t nb0,
        const uint64_t nb1,
        const uint64_t nb2) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_elem = n_embd * n_hc * n_tokens;
    if (idx >= n_elem) {
        return;
    }

    const int64_t d      = idx % n_embd;
    const int64_t tmp    = idx / n_embd;
    const int64_t dst_hc = tmp % n_hc;
    const int64_t t      = tmp / n_hc;

    const float block_v = *(const float *)(block_out + d * nb_block0 + t * nb_block1);
    const float post_v  = *(const float *)(post + dst_hc * nb_post0 + t * nb_post1);

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < n_hc; ++src_hc) {
        const float comb_v = *(const float *)(comb + dst_hc * nb_comb0 + src_hc * nb_comb1 + t * nb_comb2);
        const float res_v  = *(const float *)(residual + d * nb_res0 + src_hc * nb_res1 + t * nb_res2);
        acc += comb_v * res_v;
    }

    *(float *)(dst + d * nb0 + dst_hc * nb1 + t * nb2) = acc;
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = dst->ne[1];
    const int64_t n_tokens = dst->ne[2];
    const int64_t n_elem   = n_embd * n_hc * n_tokens;

    cudaStream_t stream = ctx.stream();

    const int num_blocks = (int)((n_elem + CUDA_DSV4_HC_EXPAND_BLOCK_SIZE - 1) / CUDA_DSV4_HC_EXPAND_BLOCK_SIZE);
    dsv4_hc_expand_kernel<<<num_blocks, CUDA_DSV4_HC_EXPAND_BLOCK_SIZE, 0, stream>>>(
        (const char *)block_out->data,
        (const char *)residual->data,
        (const char *)post->data,
        (const char *)comb->data,
        (char *)dst->data,
        n_embd, n_hc, n_tokens,
        block_out->nb[0], block_out->nb[1],
        residual->nb[0], residual->nb[1], residual->nb[2],
        post->nb[0], post->nb[1],
        comb->nb[0], comb->nb[1], comb->nb[2],
        dst->nb[0], dst->nb[1], dst->nb[2]);
}
