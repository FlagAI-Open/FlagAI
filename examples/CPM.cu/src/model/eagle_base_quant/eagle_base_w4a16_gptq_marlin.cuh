#pragma once
#include "../w4a16_gptq_marlin/w4a16_gptq_marlin_model.cuh"
#include "../eagle.cuh"

template<typename T>
struct EagleImplBaseW4A16GPTQMarlin : Model {
    int num_layers;
    int num_iter;
    int topk_per_iter;
    int tree_size;
    int total_tried;

    W4A16GPTQMarlinModelImpl<T>* model;
    KVCacheManager<T>* kv_caches;
    std::vector<Layer<T>*> layers;
    Linear<T, true, true> *fc1;
    Linear<T> *fc2;
    functions::TopK<T>* topk_func;
    functions::TopK<T>* topk_func_2;

    T *prev_hidden_state, *prev_embed;
    int num_prev, num_history_tokens;
    int32_t *eagle_position_ids, *eagle_cache_length;
    int *eagle_original_length, eagle_padded_length;
    uint64_t *eagle_mask_2d, *tmp_mask_2d;
    T* eagle_logits;
    T* tried_history_val; int32_t* tried_history_pos;
    int32_t* tried_history_parent;
    bool is_first_draft;

    int32_t *h_best, *d_best;    

    T* tmp_kvcache;

    EagleImplBaseW4A16GPTQMarlin(
        W4A16GPTQMarlinModelImpl<T>* model,
        int num_layers,
        int num_iter,
        int topk_per_iter,
        int tree_size
    ) {
        this->model = model;
        this->num_layers = num_layers;
        this->num_iter = num_iter;
        this->topk_per_iter = topk_per_iter;
        this->tree_size = tree_size;
        this->total_tried = topk_per_iter * topk_per_iter * (num_iter - 1) + topk_per_iter;

        kv_caches = new KVCacheManager<T>(num_layers, this->model->num_key_value_heads, this->model->head_dim);
        fc1 = new Linear<T, true, true>(this->model->hidden_size, this->model->hidden_size);
        fc2 = new Linear<T>(this->model->hidden_size, this->model->hidden_size);
        for (int i = 0; i < num_layers; i++) {
            layers.push_back(new Layer<T>(this->model->hidden_size, this->model->intermediate_size, this->model->num_attention_heads, this->model->num_key_value_heads, this->model->head_dim, this->model->rms_norm_eps));
        }

        assert(topk_per_iter <= this->tree_size-1);

        topk_func = new functions::TopK<T>(model->vocab_size, topk_per_iter);
        topk_func_2 = new functions::TopK<T>(total_tried, this->tree_size-1);
    }

    void init_weight_ptr(Memory* memory) {
        fc1->init_weight_ptr(memory);
        fc2->init_weight_ptr(memory);
        for (int i = 0; i < num_layers; i++) {
            layers[i]->init_weight_ptr(memory);
        }
        layers[0]->attn->attn_norm = new Skip<T>(this->model->hidden_size);
        kv_caches->rotary_embedding = this->model->kv_caches->rotary_embedding;
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        offset = fc1->init_output_ptr(memory, num_tokens, offset);
        offset = fc2->init_output_ptr(memory, num_tokens, offset);
        int64_t layer_end = 0;
        for (int i = 0; i < num_layers; i++) {
            layer_end = layers[i]->init_output_ptr(memory, num_tokens, offset);
        }
        offset = layer_end;
        offset = memory->allocate((void**)&eagle_logits, offset, this->topk_per_iter * this->model->vocab_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tmp_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tried_history_val, offset, this->total_tried * sizeof(T));
        offset = memory->allocate((void**)&tried_history_pos, offset, this->total_tried * sizeof(int32_t));
        offset = memory->allocate((void**)&tried_history_parent, offset, this->topk_per_iter * (this->num_iter - 1) * sizeof(int32_t));
        cudaMallocHost(&eagle_original_length, sizeof(int32_t));

        offset = topk_func->init_output_ptr(memory, this->topk_per_iter, offset);
        offset = topk_func_2->init_output_ptr(memory, 1, offset);

        offset = memory->allocate((void**)&prev_hidden_state, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&prev_embed, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&eagle_cache_length, offset, sizeof(int32_t));

        offset = memory->allocate((void**)&d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&h_best, 2 * sizeof(int32_t));
        offset = memory->allocate((void**)&tmp_kvcache, offset, 64 * this->model->kv_caches->num_hidden_layers * 2 * this->model->kv_caches->dim * sizeof(T));
        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);
        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = this->init_output_ptr(this->model->memory, this->model->chunk_length, offset);
        float ratio = float(this->model->num_hidden_layers) / (this->model->num_hidden_layers + this->num_layers);
        kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        kv_caches->init_output_ptr(this->model->memory, kv_cache_offset);
        return min(kv_caches->budget + 1, this->model->kv_caches->budget);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 5) == "eagle") {
            if (name.substr(0, 9) == "eagle.fc1") {
                fc1->load_to_storage(name, ptr);
            } else if (name.substr(0, 9) == "eagle.fc2") {
                fc2->load_to_storage(name, ptr);
            } else {
                std::regex layer_regex("eagle\\.layers\\.(\\d+)\\.(.*)");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    int layer_idx = std::stoi(matches[1]);
                    layers[layer_idx]->load_to_storage(matches[2], ptr);
                } else {
                    throw std::invalid_argument("Unsupported name (layer_idx not found): " + name);
                }
            }
        } else {
            this->model->load_to_storage(name, ptr);
        }
    }

    void eagle_prefill(int num_history_tokens) {
        cudaMemcpy(this->prev_embed + (num_prev - 1) * this->model->hidden_size, this->model->embedding->output, this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        this->fc1->prefill(calc_stream, num_prev, this->prev_embed);
        this->fc2->prefill(calc_stream, num_prev, this->prev_hidden_state);
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc1->output, this->fc2->output, this->fc2->output);
        T* layer_output = nullptr;
        for (int i = 0; i < num_layers; i++) {
            this->layers[i]->prefill(num_prev, num_history_tokens, this->fc2->output, layer_output, this->eagle_position_ids, this->kv_caches->caches[i]);
            layer_output = this->layers[i]->output;
        }
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc2->output, layer_output, this->fc2->output);
    }

    void eagle_decode(int32_t* cache_length) {
        this->fc1->prefill(calc_stream, num_prev, this->prev_embed);
        this->fc2->prefill(calc_stream, num_prev, this->prev_hidden_state);
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc1->output, this->fc2->output, this->fc2->output);
        T* layer_output = nullptr;
        for (int i = 0; i < num_layers; i++) {
            this->layers[i]->decode(num_prev, this->eagle_padded_length, this->fc2->output, layer_output, this->eagle_position_ids, cache_length, Mask(nullptr), this->kv_caches->caches[i]);
            layer_output = this->layers[i]->output;
        }
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc2->output, layer_output, this->fc2->output);
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->embedding->prefill(calc_stream, num_tokens, input);
        if (num_history_tokens > 0) {
            this->eagle_prefill(this->num_history_tokens);
        }

        cudaMemcpy(this->prev_embed, this->model->embedding->output + this->model->hidden_size, (num_tokens - 1) * this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        this->model->prefill_embed(num_tokens, num_history_tokens, this->model->embedding->output, position_ids, output);
        this->prev_hidden_state = this->model->norm->output;
        cudaMemcpy(this->eagle_position_ids, position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->num_prev = num_tokens;

        this->num_history_tokens = num_history_tokens;
        this->is_first_draft = true;
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->decode(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output);
    }

    void draft(int32_t* tree_draft_ids, int32_t* tree_position_ids, int32_t* cache_length, uint64_t* tree_attn_mask, int32_t* tree_parent) {
        cudaMemcpy(this->eagle_original_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
        this->eagle_padded_length = (this->eagle_original_length[0] + 256 - 1) / 128 * 128;


        if (this->is_first_draft) {
            this->model->embedding->prefill(calc_stream, 1, tree_draft_ids);
            this->eagle_prefill(this->num_history_tokens);
        } else {
            this->eagle_decode(cache_length);
        }
        cudaMemcpy(this->eagle_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->eagle_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        repeat(calc_stream, topk_per_iter, 1, 0, this->eagle_position_ids);

        { // d = 0
            this->model->lm_head->prefill(calc_stream, 1, this->fc2->output + (num_prev - 1) * this->model->hidden_size, this->eagle_logits);
            log_softmax(calc_stream, 1, this->model->vocab_size, this->eagle_logits);
            this->topk_func->prefill(calc_stream, 1, this->eagle_logits);
            cudaMemcpy(this->topk_func_2->topk_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->topk_func_2->topk_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tried_history_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tried_history_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            repeat(calc_stream, topk_per_iter, this->model->hidden_size, num_prev-1, this->fc2->output, this->fc1->output);
            init_tree(calc_stream, topk_per_iter, this->eagle_mask_2d);
        }
        for (int d = 1; d < this->num_iter; ++d) {
            add(calc_stream, 1, this->eagle_cache_length, topk_per_iter);
            this->model->embedding->prefill(calc_stream, topk_per_iter, this->topk_func_2->topk_pos);
            this->fc2->prefill(calc_stream, topk_per_iter, this->fc1->output);
            this->fc1->prefill(calc_stream, topk_per_iter, this->model->embedding->output);
            elementwise_add(calc_stream, topk_per_iter, this->model->hidden_size, this->fc1->output, this->fc2->output, this->fc2->output);
            T* layer_output = nullptr;
            for (int i = 0; i < num_layers; i++) {
                this->layers[i]->decode(topk_per_iter, this->eagle_padded_length, this->fc2->output, layer_output, this->eagle_position_ids, this->eagle_cache_length, Mask(eagle_mask_2d, topk_per_iter, topk_per_iter * d), this->kv_caches->caches[i]);
                layer_output = this->layers[i]->output;
            }
            elementwise_add(calc_stream, topk_per_iter, this->model->hidden_size, this->fc2->output, layer_output, this->fc2->output);
            add(calc_stream, topk_per_iter, this->eagle_position_ids, 1);

            this->model->lm_head->prefill(calc_stream, topk_per_iter, this->fc2->output, this->eagle_logits);
            log_softmax(calc_stream, topk_per_iter, this->model->vocab_size, this->eagle_logits);
            this->topk_func->prefill(calc_stream, topk_per_iter, this->eagle_logits);
            cumsum(calc_stream, topk_per_iter, topk_per_iter, this->topk_func->topk_val, this->topk_func_2->topk_val);
            cudaMemcpy(this->tried_history_val + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_val, topk_per_iter * topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tried_history_pos + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_pos, topk_per_iter * topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->topk_func_2->prefill(calc_stream, 1, this->topk_func->topk_val, topk_per_iter * topk_per_iter, topk_per_iter);

            cudaMemcpy(this->tmp_mask_2d, this->eagle_mask_2d, topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, topk_per_iter, this->tried_history_parent + (d - 1) * topk_per_iter, this->topk_func_2->topk_pos, topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter);
            update_tree(calc_stream, topk_per_iter, topk_per_iter * d, this->eagle_mask_2d, this->tmp_mask_2d, this->topk_func_2->topk_pos);
            remap_hidden(calc_stream, topk_per_iter, this->model->hidden_size, this->topk_func_2->topk_pos, this->fc2->output, this->fc1->output, topk_per_iter);
            remap_id(calc_stream, topk_per_iter, this->topk_func_2->topk_pos, this->topk_func->topk_pos);
        }

        this->topk_func_2->prefill(calc_stream, 1, this->tried_history_val);

        // build tree
        build_dynamic_tree(calc_stream, this->tree_size, this->eagle_original_length[0], this->topk_per_iter, this->tried_history_parent, this->topk_func_2->topk_pos, tree_position_ids, tree_attn_mask, tree_parent);
        remap_id(calc_stream, this->tree_size-1, this->topk_func_2->topk_pos, this->tried_history_pos, tree_draft_ids + 1);

        this->is_first_draft = false;
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, int32_t* tree_parent) {
        verify_draft(calc_stream, num_tokens, pred, gt, position_ids, cache_length, mask_2d, tree_parent, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);

        this->num_prev = h_best[0];
        remap_hidden(calc_stream, this->num_prev, this->model->hidden_size, pred, this->model->norm->output, this->prev_hidden_state);

        fix_kv_cache(calc_stream, h_best[0], this->model->kv_caches->num_hidden_layers * 2, this->model->kv_caches->dim, pred, gt, cache_length, this->model->kv_caches->d_flat_caches, this->tmp_kvcache);

        this->model->embedding->prefill(calc_stream, this->num_prev, pred);
        cudaMemcpy(this->prev_embed, this->model->embedding->output, this->num_prev * this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        make_arange(calc_stream, this->num_prev, cache_length, this->eagle_position_ids);

        return h_best[0];
    }
};