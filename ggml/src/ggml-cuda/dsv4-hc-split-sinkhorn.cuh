#include "common.cuh"

#define CUDA_DSV4_HC_SINKHORN_BLOCK_SIZE 256

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
