#pragma once
#include "../trait.cuh"
#include "rotary.cuh"
#include <vector>
#include <cuda_runtime.h>

template <typename T>
struct KVCache {
    int dim;
    T *k_cache, *v_cache;
    RotaryEmbedding<T> *rotary_embedding;
    
    KVCache(int dim, RotaryEmbedding<T> *rotary_embedding) {
        this->dim = dim;
        this->rotary_embedding = rotary_embedding;
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        offset = memory->allocate((void**)&this->k_cache, offset, num_tokens * dim * sizeof(T));
        offset = memory->allocate((void**)&this->v_cache, offset, num_tokens * dim * sizeof(T));
        return offset;
    }

    T* offset_k(int offset) { return k_cache + offset * dim; }
    T* offset_v(int offset) { return v_cache + offset * dim; }
};

template <typename T>
struct KVCacheManager {
    int num_hidden_layers;
    int dim;
    int budget;
    std::vector<KVCache<T>*> caches;
    T **h_flat_caches, **d_flat_caches;
    RotaryEmbedding<T> *rotary_embedding;

    KVCacheManager(int num_hidden_layers, int num_key_value_heads, int head_dim) {
        this->num_hidden_layers = num_hidden_layers;
        this->dim = num_key_value_heads * head_dim;
        this->rotary_embedding = new RotaryEmbedding<T>(head_dim);
    }

    void init_weight_ptr(Memory* memory) {
        this->rotary_embedding->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int64_t offset, float ratio=1.0) {
        offset = memory->allocate((void**)&this->d_flat_caches, offset, num_hidden_layers * 2 * sizeof(T*));

        budget = int64_t(memory->get_remaining_memory(offset) * ratio * 0.999) / (this->num_hidden_layers * 2 * this->dim * sizeof(T)) - 1;
        for (int i = 0; i < this->num_hidden_layers; i++) {
            caches.push_back(new KVCache<T>(this->dim, this->rotary_embedding));
        }
        for (int i = 0; i < this->num_hidden_layers; i++) {
            offset = caches[i]->init_output_ptr(memory, budget, offset);
        }
        this->h_flat_caches = new T*[num_hidden_layers * 2];
        for (int i = 0; i < num_hidden_layers; i++) {
            this->h_flat_caches[i * 2] = caches[i]->k_cache;
            this->h_flat_caches[i * 2 + 1] = caches[i]->v_cache;
        }
        cudaMemcpy(this->d_flat_caches, this->h_flat_caches, num_hidden_layers * 2 * sizeof(T*), cudaMemcpyHostToDevice);
        return offset;
    }
};
