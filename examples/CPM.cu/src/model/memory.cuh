#pragma once
#include "../utils.cuh"
#include <cuda_runtime.h>
#include "../signal_handler.cuh"
#ifdef DISABLE_MEMPOOL
#include <vector>
#endif

#define ALIGN_SIZE 256

// TODO: refactor this for better encapsulation
struct Memory {
    int64_t memory_limit;
    int64_t model_offset;
#ifndef DISABLE_MEMPOOL
    uint8_t* memory_pool;
#else
    int64_t allocated_size;
    std::vector<void*> allocated_ptrs;
#endif

    Memory(float memory_limit) {
        // Get GPU total memory size
        size_t free_memory, total_memory;
        cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: cudaMemGetInfo failed: %s\n\n", cudaGetErrorString(err));
            this->memory_limit = 0;
            this->model_offset = 0;
#ifndef DISABLE_MEMPOOL
            this->memory_pool = nullptr;
#else
            this->allocated_size = 0;
#endif
            return;
        }
        
        // Calculate actual memory size
        this->memory_limit = (int64_t)(total_memory * memory_limit);

#ifndef DISABLE_MEMPOOL
        printf("Use Pre-allocated Memory Pool\n");
#else
        printf("Use Dynamic Memory Allocation, this is for debug\n");
#endif
        printf("GPU Total Memory: %ld bytes (%.2f GB), ", total_memory, (double)total_memory / (1024*1024*1024));
        printf("Set Allocatable Memory Limit: %ld bytes (%.2f GB), ratio: %.1f%%\n",
               this->memory_limit, (double)this->memory_limit / (1024*1024*1024), memory_limit * 100);
        
        this->model_offset = 0;
#ifndef DISABLE_MEMPOOL
        err = cudaMalloc(reinterpret_cast<void**>(&this->memory_pool), this->memory_limit);
        if (err != cudaSuccess) {
            print_stack_trace(5);
            fprintf(stderr, "Error: cudaMalloc failed in Memory constructor: %s, size: %ld\n\n", cudaGetErrorString(err), this->memory_limit);
            this->memory_pool = nullptr;
        }
#else
        // In DISABLE_MEMPOOL mode, don't pre-allocate memory
        this->allocated_size = 0;
#endif
    }

    // Add destructor to prevent memory leak
    ~Memory() {
#ifndef DISABLE_MEMPOOL
        if (memory_pool != nullptr) {
            cudaError_t err = cudaFree(memory_pool);
            if (err != cudaSuccess) {
                fprintf(stderr, "Warning: cudaFree failed in Memory destructor: %s\n\n", cudaGetErrorString(err));
            }
        }
#else
        // In DISABLE_MEMPOOL mode, free all individually allocated memory
        for (void* ptr : allocated_ptrs) {
            if (ptr != nullptr) {
                cudaError_t err = cudaFree(ptr);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Warning: cudaFree failed in Memory destructor: %s\n\n", cudaGetErrorString(err));
                }
            }
        }
        allocated_ptrs.clear();
#endif
    }

    // Get remaining available memory from a specific offset
    int64_t get_remaining_memory(int64_t offset) const {
#ifndef DISABLE_MEMPOOL
        return this->memory_limit - offset;
#else
        return this->memory_limit - this->allocated_size;
#endif
    }

#ifndef DISABLE_MEMPOOL
    void* allocate_for_model(size_t size) {
        uint8_t* ret = memory_pool + model_offset;
        model_offset += size;
        model_offset = ROUND_UP(model_offset, ALIGN_SIZE);
        if (model_offset > this->memory_limit) {
            print_stack_trace(5);
            fprintf(stderr, "Error: memory limit exceeded, offset: %ld, size: %ld, memory_limit: %ld\n\n", model_offset, size, this->memory_limit);
            return nullptr;
        }
        return (void*)ret;
    }
    int64_t allocate(void** ptr, int64_t offset, size_t size = 0) {
        if (size == 0) {
            print_stack_trace(5);
            fprintf(stderr, "Error: size is 0\n\n");
            return -1;
        }
        *ptr = memory_pool + offset;
        offset += size;
        offset = ROUND_UP(offset, ALIGN_SIZE);
        if (offset > this->memory_limit) {
            print_stack_trace(5);
            fprintf(stderr, "Error: memory limit exceeded, offset: %ld, size: %ld, memory_limit: %ld\n\n", offset, size, this->memory_limit);
            *ptr = nullptr;
            return -1;
        }
        return offset;
    }
#else
    void* allocate_for_model(size_t size) {
        void* ptr;
        size_t aligned_size = ROUND_UP(size, ALIGN_SIZE);
        
        // Check if allocation would exceed memory limit
        if (allocated_size + aligned_size > this->memory_limit) {
            print_stack_trace(5);
            fprintf(stderr, "Error: memory limit exceeded, allocated_size: %ld, new_size: %ld, memory_limit: %ld\n\n", 
                    allocated_size, aligned_size, this->memory_limit);
            return nullptr;
        }
        
        cudaError_t err = cudaMalloc(&ptr, aligned_size);
        if (err != cudaSuccess) {
            print_stack_trace(5);
            fprintf(stderr, "Error: cudaMalloc failed: %s, size: %ld\n\n", cudaGetErrorString(err), size);
            return nullptr;
        }
        
        allocated_ptrs.push_back(ptr);
        allocated_size += aligned_size;
        return ptr;
    }
    int64_t allocate(void** ptr, int64_t offset, size_t size = 0) { // 0 for reuse previous allocated memory, just need start offset, return value is useless
        if (size == 0) {
            print_stack_trace(5);
            fprintf(stderr, "Error: size is 0\n\n");
            return -1;
        }
        
        size_t aligned_size = ROUND_UP(size, ALIGN_SIZE);
        
        // Check if allocation would exceed memory limit
        if (allocated_size + aligned_size > this->memory_limit) {
            print_stack_trace(5);
            fprintf(stderr, "Error: memory limit exceeded, allocated_size: %ld, new_size: %ld, memory_limit: %ld\n\n", 
                    allocated_size, aligned_size, this->memory_limit);
            *ptr = nullptr;
            return -1;
        }
        
        cudaError_t err = cudaMalloc(ptr, aligned_size);
        if (err != cudaSuccess) {
            print_stack_trace(5);
            fprintf(stderr, "Error: cudaMalloc failed: %s, size: %ld\n\n", cudaGetErrorString(err), size);
            *ptr = nullptr;
            return -1;
        }
        
        allocated_ptrs.push_back(*ptr);
        allocated_size += aligned_size;
        
        // Update max_output_offset for tracking purposes
        offset += aligned_size;
        return offset;
    }
#endif
};