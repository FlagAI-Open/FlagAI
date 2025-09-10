#pragma once
#include "../utils.cuh"
#include "../trait.cuh"
namespace functions {
namespace {
template<typename T, int N>
static __device__ inline void warpBitonicSort(T& v1, int& pos, bool asc) {
    int lane_id = threadIdx.x & (N - 1);
    #pragma unroll
    for (int k = 2; k <= N; k *= 2) {
        bool desc = ((lane_id & k) == 0) ^ asc;
        #pragma unroll
        for (int j = k / 2; j > 0; j /= 2) {
            T v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
            int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos, j);
            bool upper = (lane_id & j) != 0;
            if (desc ^ (v1 > v2 || (v1 == v2 && pos < pos2)) ^ upper) {
                v1 = v2;
                pos = pos2;
            }
        }
    }
}
template<typename T, int N>
static __device__ inline void warpBitonicMerge(T& v1, int& pos1, T& v2, int& pos2) {
    if (v1 < v2 || (v1 == v2 && pos1 > pos2)) {
        v1 = v2;
        pos1 = pos2;
    }
    int lane_id = threadIdx.x & (N - 1);
    // resort
    #pragma unroll
    for (int j = N / 2; j > 0; j /= 2) {
        v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
        int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos1, j);
        bool upper = (lane_id & j) != 0;
        if ((v1 < v2 || (v1 == v2 && pos1 > pos2)) ^ upper) {
            v1 = v2;
            pos1 = pos2;
        }
    }
}
template<typename T, int N>
static __device__ inline void blockBitonicReduce(T& v, int& pos) {
    __shared__ T shared_val[1024];
    __shared__ int shared_pos[1024];
    // block reduce
    shared_val[threadIdx.x] = v;
    shared_pos[threadIdx.x] = pos;
    // inter warp reduce
    #pragma unroll
    for (int i = 512; i >= 32; i >>= 1) {
        if (blockDim.x > i) {
            __syncthreads();
            if (threadIdx.x < i) {
                int idx_next = (i << 1) - threadIdx.x - 1;
                T nw_v = (idx_next < blockDim.x) ? shared_val[idx_next] : T(-TypeTraits<T>::inf());
                int nw_pos = (idx_next < blockDim.x) ? shared_pos[idx_next] : -1;
                warpBitonicMerge<T, N>(v, pos, nw_v, nw_pos); // merge and rebuild in desc order
                shared_val[threadIdx.x] = v;
                shared_pos[threadIdx.x] = pos;
            }
        }
    }
    // intra warp reduce
    if (threadIdx.x < 32) {
        warpBitonicSort<T, 32>(v, pos, false);
    }
}
template<typename T, int N>
static __global__ void kernel_bitonic_topk(
    int n, int top,
    T *inp,     // (batch, n)
    float *out,     // (batch, top)
    int *idx    // (batch, top)
) {
    int offset_inp = blockIdx.x * n;
    int offset_out = blockIdx.x * top;
    T local_v = threadIdx.x < n ? inp[offset_inp + threadIdx.x] : -TypeTraits<T>::inf();
    int local_pos = threadIdx.x;
    warpBitonicSort<T, N>(local_v, local_pos, false); // local sort in desc order
    for (int i = blockDim.x; i < n; i += blockDim.x) {
        T nw_v = (i + threadIdx.x) < n ? inp[offset_inp + i + threadIdx.x] : -TypeTraits<T>::inf();
        int nw_pos = i + threadIdx.x;
        // step.1: local sort
        warpBitonicSort<T, N>(nw_v, nw_pos, true); // local sort in asc order
        // step.2&3: merge and rebuild
        warpBitonicMerge<T, N>(local_v, local_pos, nw_v, nw_pos); // merge and rebuild in desc order
    }
    blockBitonicReduce<T, N>(local_v, local_pos);
    if (threadIdx.x < top) {
        out[offset_out + threadIdx.x] = local_v;
        idx[offset_out + threadIdx.x] = local_pos;
    }
}
// intra-block topk
// gridDim(batch, n / 1024, 1), threadDim(1024, 1, 1)
template<typename T, int N, bool ordered>
static __global__ void kernel_bitonic_topk_multiblock(
    int n,
    const T *inp,       // (batch, n)
    const int *idx_inp, // (batch, n)
    T *out,     // (batch, n / 1024 * N)
    int *idx    // (batch, n / 1024 * N)
) {
    int offset_col = blockIdx.y * blockDim.x + threadIdx.x;
    int offset_inp = blockIdx.x * n + offset_col;
    int offset_out = blockIdx.x * (gridDim.y * N) + blockIdx.y * N + threadIdx.x;
    T local_v = (offset_col < n) ? inp[offset_inp] : T(-TypeTraits<T>::inf());
    int local_pos = (idx_inp == nullptr) ? offset_col : ((offset_col < n) ? idx_inp[offset_inp] : -1);
    if (!ordered) warpBitonicSort<T, N>(local_v, local_pos, false); // local sort in desc order
    blockBitonicReduce<T, N>(local_v, local_pos);
    if (threadIdx.x < N) {
        out[offset_out] = local_v;
        idx[offset_out] = local_pos;
    }
}
// copy kernel
// gridDim(batch, 1, 1),   blockDim(top, 1, 1)
template<typename T>
static __global__ void kernel_bitonic_topk_multiblock_copy (
    int n, int top,
    const T *inp,       // (batch, n)
    const int *idx_inp, // (batch, n)
    T *out,         // (batch, top)
    int *idx            // (batch, top)
) {
    int offset_inp = blockIdx.x * n + threadIdx.x;
    int offset_out = blockIdx.x * top + threadIdx.x;
    if (threadIdx.x < top) {
        out[offset_out] = inp[offset_inp];
        idx[offset_out] = idx_inp[offset_inp];
    }
}
#define TOPK_SIZE_DISPATCH(top, ...) \
    do { \
        const int &top_v = top; \
        if (top_v > 16) { \
            const int top_size = 32; \
            __VA_ARGS__ \
        } else if (top_v > 8) { \
            const int top_size = 16; \
            __VA_ARGS__ \
        } else if (top_v > 4) { \
            const int top_size = 8; \
            __VA_ARGS__ \
        } else if (top_v > 2) { \
            const int top_size = 4; \
            __VA_ARGS__ \
        } else if (top_v > 1) { \
            const int top_size = 2; \
            __VA_ARGS__ \
        } else { \
            const int top_size = 1; \
            __VA_ARGS__ \
        } \
    } while(0)
template <typename T>
void bitonic_topk(
    const Stream& stream,
    const int batch,
    const int n,
    const int top,
    const T* x, 
    T* out, 
    int* pos,	
    T* buf_val,
    int* buf_pos,
    T* nw_buf_val,
    int* nw_buf_pos
) {
    TOPK_SIZE_DISPATCH(top, {
        bool first = true;
        dim3 blockDim(1024, 1, 1);
        unsigned int tmp_n = n;
        do {
            dim3 gridDim(batch, CEIL_DIV(tmp_n, 1024), 1);
            if (first) {
                first = false;
                kernel_bitonic_topk_multiblock<T, top_size, false><<<gridDim, blockDim, 0, stream.stream>>>(
                    tmp_n,
                    x,
                    nullptr,
                    buf_val,
                    buf_pos
                );
            } else {
                kernel_bitonic_topk_multiblock<T, top_size, false><<<gridDim, blockDim, 0, stream.stream>>>(
                    tmp_n,
                    buf_val,
                    buf_pos,
                    nw_buf_val,
                    nw_buf_pos
                );
                buf_val = nw_buf_val;
                buf_pos = nw_buf_pos;
            }
            tmp_n = CEIL_DIV(tmp_n, 1024) * top_size;
        } while (tmp_n > top_size);
        // copy to output tensor
        {
            dim3 gridDim(batch, 1, 1);
            blockDim = dim3(top_size, 1, 1);
            kernel_bitonic_topk_multiblock_copy<T><<<gridDim, blockDim, 0, stream.stream>>>(
                top_size, top,
                buf_val,
                buf_pos,
                out,
                pos
            );
        }
    });
}

template<typename T>
static __global__ void set_topk_to_neg_inf_kernel(int dim, int top, T* x, const int* topk_pos) {
    x[blockIdx.x * dim + topk_pos[blockIdx.x * top + threadIdx.x]] = -TypeTraits<T>::inf();
}
} // namespace

template<typename T>
void set_topk_to_neg_inf(const Stream& stream, int num_tokens, int dim, int top, int num, T* x, const int* topk_pos) {
    set_topk_to_neg_inf_kernel<<<num_tokens, num, 0, stream.stream>>>(dim, top, x, topk_pos);
}

template <typename T>
struct TopK {
private:
    T *buf_val, *nw_buf_val;
    int *buf_pos, *nw_buf_pos;
public:
    int dim, top;
    T* topk_val;
    int* topk_pos;
    T* tmp_x;

    TopK(const int dim, const int top) {
        this->dim = dim;
        this->top = top;
    }
    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        TOPK_SIZE_DISPATCH(top, {
            offset = memory->allocate((void**)&buf_val, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(T));
            offset = memory->allocate((void**)&buf_pos, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(int));
            offset = memory->allocate((void**)&nw_buf_val, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(T));
            offset = memory->allocate((void**)&nw_buf_pos, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(int));
        });
        if (top > 32) { // tmp fix
            offset = memory->allocate((void**)&tmp_x, offset, num_tokens * dim * sizeof(T));
        }
        offset = memory->allocate((void**)&topk_val, offset, num_tokens * top * sizeof(T));
        offset = memory->allocate((void**)&topk_pos, offset, num_tokens * top * sizeof(int));
        return offset;
    }
    void prefill(
        const Stream& stream,
        int num_tokens,
        const T* input,
        int dim = -1,
        int top = -1
    ) {
        assert(dim == -1 || dim <= this->dim);
        assert(top == -1 || top <= this->top);
        if (dim == -1) dim = this->dim;
        if (top == -1) top = this->top;
        bitonic_topk<T>(
            stream,
            num_tokens,
            dim, top,
            input,
            this->topk_val, this->topk_pos,
            this->buf_val, this->buf_pos,
            this->nw_buf_val, this->nw_buf_pos
        );
        if (top > 32) {
            for (int i = 32; i < top; i += 32) {
                cudaMemcpyAsync(this->tmp_x, input, num_tokens * dim * sizeof(T), cudaMemcpyDeviceToDevice, stream.stream);
                set_topk_to_neg_inf(stream, num_tokens, dim, top, 32, this->tmp_x, this->topk_pos + (i - 32));
                bitonic_topk<T>(
                    stream,
                    num_tokens,
                    dim, top,
                    this->tmp_x,
                    this->topk_val + i, this->topk_pos + i,
                    this->buf_val, this->buf_pos,
                    this->nw_buf_val, this->nw_buf_pos
                );
            }
        }
    }
};
} // namespace functions