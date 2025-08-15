#pragma once

#include <cuda_runtime.h>

struct Mask {
    uint64_t* ptr;
    int mask_q_range;
    int mask_k_range;

    Mask(uint64_t* ptr = nullptr, int mask_q_range = 0, int mask_k_range = 0) : ptr(ptr) {
        if (ptr == nullptr) {
            mask_q_range = 0;
            mask_k_range = 0;
        }
        this->mask_q_range = mask_q_range;
        this->mask_k_range = mask_k_range;
    }
};
