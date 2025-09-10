namespace {
const int WARP_SZ = 32;

// blocks <block_size>,      threads<1024>
__device__ float block_reduce_sum(float val) {
    static __shared__ float s_x[WARP_SZ];
    // int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int lid = threadIdx.x % WARP_SZ;
    int wid = threadIdx.x / WARP_SZ;

    // reduce intra warp

    for (int offset = WARP_SZ/2; offset > 0; offset >>= 1) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    
    if (lid == 0) s_x[wid] = val;
    __syncthreads();

    // reduce inter warp
    val = (tid < WARP_SZ) ? s_x[lid] : 0;
    if (wid == 0) {
        for (int offset = WARP_SZ/2; offset > 0; offset >>= 1) 
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// blocks <block_size>,      threads<1024>
__device__ float block_reduce_max(float val) {
    static __shared__ float s_x[WARP_SZ];
    // int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int lid = threadIdx.x % WARP_SZ;
    int wid = threadIdx.x / WARP_SZ;

    // reduce intra warp

    for (int offset = WARP_SZ/2; offset > 0; offset >>= 1) 
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    
    if (lid == 0) s_x[wid] = val;
    __syncthreads();

    // reduce inter warp
    val = (tid < WARP_SZ) ? s_x[lid] : -INFINITY;
    if (wid == 0) {
        for (int offset = WARP_SZ/2; offset > 0; offset >>= 1) 
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// blocks <block_size>,      threads<1024>
__device__ float block_allreduce_sum(float val) {
    static __shared__ float s_x[WARP_SZ];
    // int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int lid = threadIdx.x % WARP_SZ;
    int wid = threadIdx.x / WARP_SZ;

    // reduce intra warp

    for (int offset = WARP_SZ/2; offset > 0; offset >>= 1) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    
    if (lid == 0) s_x[wid] = val;
    __syncthreads();

    // reduce inter warp
    val = (tid < WARP_SZ) ? s_x[lid] : 0;
    if (wid == 0) {
        for (int offset = WARP_SZ/2; offset > 0; offset >>= 1) 
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (tid == 0) {
        s_x[0] = val;
    }
    __syncthreads();
    return s_x[0];
}

// blocks <block_size>,      threads<1024>
__device__ float block_allreduce_max(float val) {
    static __shared__ float s_x[WARP_SZ];
    // int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int lid = threadIdx.x % WARP_SZ;
    int wid = threadIdx.x / WARP_SZ;

    // reduce intra warp

    for (int offset = WARP_SZ/2; offset > 0; offset >>= 1) 
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    
    if (lid == 0) s_x[wid] = val;
    __syncthreads();

    // reduce inter warp
    val = (tid < WARP_SZ) ? s_x[lid] : -INFINITY;
    if (wid == 0) {
        for (int offset = WARP_SZ/2; offset > 0; offset >>= 1) 
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }

    if (tid == 0) {
        s_x[0] = val;
    }
    __syncthreads();
    return s_x[0];
}

}