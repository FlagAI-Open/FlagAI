#pragma once
#include "tree_drafter.cuh"
#include "model.cuh"
#include "topk.cuh"

namespace {
__global__ void build_tree_kernel(int tree_size, const int32_t* topk_pos, const int32_t* tree_indices, const int32_t* draft_position_ids, int32_t* tree_draft_ids, int32_t* tree_position_ids, const int32_t* cache_length) {
    int i = threadIdx.x;
    if (i < tree_size) {
        tree_position_ids[i] = cache_length[0] + draft_position_ids[i];
        if (i > 0) {
            tree_draft_ids[i] = topk_pos[tree_indices[i-1]];
        }
    }
}
}

void build_tree(const Stream& stream, int tree_size, const int32_t* topk_pos, const int32_t* tree_indices, const int32_t* draft_position_ids, int32_t* tree_draft_ids, int32_t* tree_position_ids, const int32_t* cache_length) {
    build_tree_kernel<<<1, 64, 0, stream.stream>>>(tree_size, topk_pos, tree_indices, draft_position_ids, tree_draft_ids, tree_position_ids, cache_length);
}

template<typename T>
struct ResidualBlock : Linear<T, /*transposed=*/true, /*bias=*/true> {
    ResidualBlock(int dim_in, int dim_out) : Linear<T, true, true>(dim_in, dim_out) {}

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* tgt=nullptr, bool inplace=false) {
        if (tgt == nullptr) tgt = this->output;
        Linear<T, true, true>::prefill(stream, num_tokens, input);
        silu_inplace<T>(stream, num_tokens, this->dim_out, this->output);
        elementwise_add<T>(stream, num_tokens, this->dim_out, this->output, input, tgt);
    }
};

template<typename T>
struct MedusaImpl : Model {
    int num_heads;
    int num_layers;
    int topk_per_head;
    int tree_size;
    int V;
    int32_t* tree_indices;
    int32_t* draft_position_ids;

    ModelImpl<T>* model;
    std::vector<ResidualBlock<T>*> blocks;
    std::vector<Linear<T>*> lm_heads;

    T* last_token_hidden_state;
    int32_t *h_best, *d_best;    
    T* logits;

    T* tmp_kvcache;
    functions::TopK<T>* topk_func;

    int32_t* token_id_remap;

    MedusaImpl(
        ModelImpl<T>* model,
        int num_heads,
        int num_layers,
        int topk_per_head,
        int tree_size,
        int V,
        int32_t* tree_indices,
        int32_t* draft_position_ids
    ) {
        this->model = model;
        this->num_heads = num_heads;
        this->num_layers = num_layers; // asserted in python that num_layers == 1
        this->topk_per_head = topk_per_head;
        this->tree_size = tree_size;
        this->tree_indices = tree_indices;
        this->draft_position_ids = draft_position_ids;
        this->V = V;

        for (int i = 0; i < num_heads; i++) {
            blocks.push_back(new ResidualBlock<T>(model->hidden_size, model->hidden_size));
            lm_heads.push_back(new Linear<T>(model->hidden_size, V));
        }
        topk_func = new functions::TopK<T>(V, topk_per_head);
    }

    void init_weight_ptr(Memory* memory) {
        for (int i = 0; i < num_heads; i++) {
            blocks[i]->init_weight_ptr(memory);
            lm_heads[i]->init_weight_ptr(memory);
        }
        token_id_remap = (int32_t*)memory->allocate_for_model(V * sizeof(int32_t));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        for (int i = 0; i < num_heads; i++) {
            offset = blocks[i]->init_output_ptr(memory, num_tokens, offset);
        } 
        offset = memory->allocate((void**)&logits, offset, this->num_heads * this->V * sizeof(T)); // lm_head
        offset = topk_func->init_output_ptr(memory, this->num_heads, offset);

        offset = memory->allocate((void**)&d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&h_best, 2 * sizeof(int32_t));
        offset = memory->allocate((void**)&tmp_kvcache, offset, 64 * this->model->kv_caches->num_hidden_layers * 2 * this->model->kv_caches->dim * sizeof(T));
        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);
        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = this->init_output_ptr(this->model->memory, 1, offset);
        this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset);
        return this->model->kv_caches->budget;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 6) == "medusa") {
            if (name.substr(0, 21) == "medusa.token_id_remap") {
                cudaMemcpy((void*)token_id_remap, ptr, V * sizeof(int32_t), cudaMemcpyHostToDevice);
                printf("HERE");
            } else {
                std::regex layer_regex("medusa\\.(\\d+)\\.(\\d+).*");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    int head_idx = std::stoi(matches[1]);
                    int layer_idx = std::stoi(matches[2]);
                    if (layer_idx == 0) {
                        blocks[head_idx]->load_to_storage(name, ptr);
                    } else {
                        T* tmp;
                        cudaMalloc(&tmp, this->model->hidden_size * this->model->vocab_size * sizeof(T));
                        cudaMemcpy((void*)tmp, ptr, this->model->hidden_size * this->model->vocab_size * sizeof(T), cudaMemcpyHostToDevice);
                        remap_copy(calc_stream, tmp, lm_heads[head_idx]->weight, this->model->hidden_size, V, this->token_id_remap);
                        cudaFree(tmp);
                    }
                }
            }
        } else {
            this->model->load_to_storage(name, ptr);
        }
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->prefill(num_tokens, num_history_tokens, input, position_ids, output);
        this->last_token_hidden_state = this->model->norm->output + (num_tokens - 1) * this->model->hidden_size;
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->decode(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output);
    }

    void draft(int32_t* tree_draft_ids, int32_t* tree_position_ids, int32_t* cache_length, uint64_t*, int32_t*) {
        for (int i = 0; i < num_heads; i++) {
            blocks[i]->prefill(calc_stream, 1, this->last_token_hidden_state);
            lm_heads[i]->prefill(calc_stream, 1, blocks[i]->output, this->logits + i * this->V);
        }
        topk_func->prefill(calc_stream, num_heads, this->logits);
        remap(calc_stream, num_heads * topk_per_head, topk_func->topk_pos, topk_func->topk_pos, this->token_id_remap);
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, int32_t* tree_parent) {
        verify_draft(calc_stream, num_tokens, pred, gt, position_ids, cache_length, mask_2d, tree_parent, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);
        fix_kv_cache(calc_stream, h_best[0], this->model->kv_caches->num_hidden_layers * 2, this->model->kv_caches->dim, pred, gt, cache_length, this->model->kv_caches->d_flat_caches, this->tmp_kvcache);
        this->last_token_hidden_state = this->model->norm->output + h_best[1] * this->model->hidden_size;
        return h_best[0];
    }
};