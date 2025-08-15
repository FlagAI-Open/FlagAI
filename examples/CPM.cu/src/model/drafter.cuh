#pragma once
#include <cuda_runtime.h>

namespace {

__global__ void seq_verify_kernel(int num_tokens, int32_t* pred, const int32_t* gt, const uint16_t* attn_mask, int32_t* d_best) {
    int i = threadIdx.x;

    __shared__ uint16_t s_correct_mask;
    uint16_t correct_mask = 1;
    if (0 < i && i < num_tokens && pred[i] == gt[i-1]) correct_mask |= 1ULL << i;
    // only 16 threads
    correct_mask |= __shfl_down_sync(0x0000ffff, correct_mask, 8);
    correct_mask |= __shfl_down_sync(0x0000ffff, correct_mask, 4);
    correct_mask |= __shfl_down_sync(0x0000ffff, correct_mask, 2);
    correct_mask |= __shfl_down_sync(0x0000ffff, correct_mask, 1);
    if (i == 0) s_correct_mask = correct_mask;
    __syncthreads();
    correct_mask = s_correct_mask;

    __shared__ int32_t mx[16];
    // int prefix_length = cache_length[0];
    if (i < num_tokens && ((correct_mask & attn_mask[i]) == attn_mask[i])) {
        mx[i] = i + 1; 
    } else {
        mx[i] = 1; 
    }
    __syncthreads();
    for (int offset = 8; offset > 0; offset >>= 1) {
        if (i < offset && mx[i + offset] > mx[i]) {
            mx[i] = mx[i + offset];
        }
        __syncthreads();
    }
    if (i == 0) {
        d_best[0] = mx[0]; 
    }
    __syncthreads();

}

}


void verify_seq_draft(const Stream& stream, int num_tokens, int32_t* pred, const int32_t* gt, const uint16_t* attn_mask, int32_t* best) {
    // each SM has 32 threads
    seq_verify_kernel<<<1, 16, 0, stream.stream>>>(num_tokens, pred, gt, attn_mask, best);
}