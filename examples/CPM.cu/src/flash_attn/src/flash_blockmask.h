#pragma once

namespace flash {

class fwdIterator{
    public:
    template<typename Params, typename BlockInfo>
    __device__ fwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max) {//row first
        this->cache_seqlen_k = binfo.actual_seqlen_k - binfo.actual_seqlen_q / params.m_block_dim;
        this->max_block_idx = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
        this->n_block_min = n_block_min;
        this->n_block_max = n_block_max;
        this->batch_idx = batch_idx;  // Store batch_idx for debugging
        this->head_idx = head_idx;

        const int q_block_idx = loop_step_idx + cache_seqlen_k;
        if (params.blockmask != nullptr) {
            // Calculate the offset for the uint64 blockmask 
            const int num_blocks_m = params.num_blocks_m;
            const int num_blocks_n = params.num_blocks_n;
            const int uint64_per_row = (num_blocks_n + 64 - 1) / 64;
            const int row_offset = params.cu_seqlens_q != nullptr ? binfo.blockmask_q_offset(params.m_block_dim, batch_idx) : batch_idx * params.h_k * params.num_blocks_m;

            blockmask_ptr = params.blockmask + 
                            head_idx * params.num_blocks_m * uint64_per_row + 
                            row_offset * uint64_per_row +
                            loop_step_idx * uint64_per_row;

            this->k_window_left = params.block_window_size > 0 ? (q_block_idx + kBlockN - 1) / kBlockN - params.block_window_size : 2147483647;
        } else {
            blockmask_ptr = nullptr;
            this->k_window_left = params.block_window_size > 0 ? (q_block_idx + kBlockN - 1) / kBlockN - params.block_window_size / kBlockN : -1;
        }
    }

    __device__ int _max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        if (target < 0) return -1;

        if (blockmask_ptr == nullptr) {
            if (k_window_left <= target) return target; // sliding window
            return -1;
        }
        
        if (k_window_left <= target) {
            return target;
        }
        
        // 目标值不能超过最大块索引
        target = min(target, max_block_idx - 1);
        
        // 计算相对于当前q_bit_position的实际位置
        int target_bit_pos = target;
        
        // 确定此块在哪个uint64中
        int uint64_offset = target_bit_pos / 64;
        
        // 确定此块在uint64中的哪一位
        int bit_pos = target_bit_pos % 64;
        
        // 创建一个掩码，保留target及更低位的所有位
        uint64_t mask = bit_pos != 63 ? (1ULL << (bit_pos + 1)) - 1 : 0xFFFFFFFFFFFFFFFFULL;
        
        // 检查当前uint64中target及以下的位
        uint64_t value = blockmask_ptr[uint64_offset] & mask;
        
        // 如果当前uint64中有设置的位
        if (value != 0) {
            // 找到最高位的1（即不大于target的最大设置位）
            int highest_bit = 63 - __clzll(value);  // __clzll计算前导0的数量
            int result = highest_bit + (uint64_offset * 64);
            return result;
        }
        
        // 如果当前uint64中没有找到，检查更低的uint64块
        for (int i = uint64_offset - 1; i >= 0; i--) {
            value = blockmask_ptr[i];
            if (value != 0) {
                // 找到最高位的1
                int highest_bit = 63 - __clzll(value);
                // 计算相对于q_bit_position的偏移
                int result = highest_bit + (i * 64);
                return result;
            }
        }
        
        // 没有找到设置位
        return -1;
    }

    __device__ int max_no_larger(int target) const {
        int res = _max_no_larger(target);
        return res < this->n_block_min ? -1 : res;
    }

    uint64_t *blockmask_ptr;
    int row_offset; // 行偏移量
    int uint64_per_row;          // 每行使用的uint64数量
    int cache_seqlen_k;
    int max_block_idx;
    int n_block_min, n_block_max;
    int batch_idx, head_idx;
    int k_window_left;
};

}  // namespace flash