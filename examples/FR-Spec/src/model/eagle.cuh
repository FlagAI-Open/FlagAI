#pragma once
#include "tree_drafter.cuh"
#include "model.cuh"
#include "topk.cuh"
#include "layer.cuh"
#include "kvcache.cuh"
#include "norm.cuh"
#include "elementwise.cuh"

namespace {
__global__ void add_kernel(int32_t* ptr, int32_t value) {
    ptr[threadIdx.x] += value;
}

__global__ void repeat_kernel(int32_t dim, int32_t pos, const float4* input, float4* output) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    for (int i = col; i < dim; i += blockDim.x) {
        output[row * dim + i] = input[pos * dim + i];
    }
}

template<typename T>
__global__ void repeat_kernel_2(int32_t pos, const T* input, T* output) {
    int col = threadIdx.x;
    output[col] = input[pos];
}

template<typename T>
__global__ void log_softmax_kernel(int32_t dim, T* input) {
    int base_idx = blockIdx.x * dim;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    __shared__ float s_val[32];
    float mx = -TypeTraits<T>::inf();
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        mx = fmaxf(float(input[base_idx + i]), mx);
    }
    mx = fmaxf(__shfl_down_sync(0xffffffff, mx, 16), mx);
    mx = fmaxf(__shfl_down_sync(0xffffffff, mx, 8), mx);
    mx = fmaxf(__shfl_down_sync(0xffffffff, mx, 4), mx);
    mx = fmaxf(__shfl_down_sync(0xffffffff, mx, 2), mx);
    mx = fmaxf(__shfl_down_sync(0xffffffff, mx, 1), mx);
    if (lane_id == 0) s_val[warp_id] = mx;
    __syncthreads();
    if (threadIdx.x < 32) {
        mx = s_val[threadIdx.x];
        mx = fmaxf(__shfl_down_sync(0x0000ffff, mx, 16), mx);
        mx = fmaxf(__shfl_down_sync(0x0000ffff, mx, 8), mx);
        mx = fmaxf(__shfl_down_sync(0x0000ffff, mx, 4), mx);
        mx = fmaxf(__shfl_down_sync(0x0000ffff, mx, 2), mx);
        mx = fmaxf(__shfl_down_sync(0x0000ffff, mx, 1), mx);
    }
    if (threadIdx.x == 0) {
        s_val[0] = mx;
    }
    __syncthreads();
    mx = s_val[0];

    float sum = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += expf(float(input[base_idx + i]) - mx);
    }
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (lane_id == 0) s_val[warp_id] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = s_val[threadIdx.x];
        sum += __shfl_down_sync(0x0000ffff, sum, 16);
        sum += __shfl_down_sync(0x0000ffff, sum, 8);
        sum += __shfl_down_sync(0x0000ffff, sum, 4);
        sum += __shfl_down_sync(0x0000ffff, sum, 2);
        sum += __shfl_down_sync(0x0000ffff, sum, 1);
    }
    if (threadIdx.x == 0) {
        s_val[0] = sum;
    }
    __syncthreads();
    sum = s_val[0];
    
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        input[base_idx + i] = T( float(input[base_idx + i]) - mx - logf(sum) );
    }
}

__global__ void init_tree_kernel(uint64_t* mask_2d) {
    mask_2d[threadIdx.x] = 1ULL << threadIdx.x;
}

__global__ void set_parent_kernel(int32_t num_tokens, int32_t* parent, const int32_t* pos, int32_t offset) {
    parent[threadIdx.x] = pos[threadIdx.x] + offset;
}

__global__ void update_tree_kernel(int32_t num_tokens, int32_t offset, uint64_t* mask_2d, const uint64_t* tmp_mask_2d, const int32_t* topk_pos) {
    mask_2d[threadIdx.x] = tmp_mask_2d[topk_pos[threadIdx.x] / num_tokens] | (1ULL << (offset + threadIdx.x));
}

template<typename T>
__global__ void cumsum_kernel(int32_t dim, T* input, const T* weight) {
    input[blockIdx.x * dim + threadIdx.x] += weight[blockIdx.x];
}

__global__ void remap_hidden_kernel(int32_t scale, int32_t dim, const int32_t* id_map, const float4* real_hidden, float4* output) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    int real_row = id_map[row] / scale;
    for (int i = col; i < dim; i += blockDim.x) {
        output[row * dim + i] = real_hidden[real_row * dim + i];
    }
}

__global__ void remap_id_kernel(const int32_t* id_map, const int32_t* real_id, const int32_t* token_id_remap, int32_t* output) {
    output[threadIdx.x] = token_id_remap[real_id[id_map[threadIdx.x]]];
}

__global__ void make_arange_kernel(int32_t* offset, int32_t* output) {
    output[threadIdx.x] = threadIdx.x + offset[0];
}
} // namespace

void add(const Stream& stream, int32_t num_tokens, int32_t* ptr, int32_t value) {
    add_kernel<<<1, num_tokens, 0, stream.stream>>>(ptr, value);
}

template<typename T>
void repeat(const Stream& stream, int32_t num_tokens, int32_t dim, int32_t pos, T* input, T* output=nullptr) {
    if (output == nullptr) output = input;
    if (dim > 1) {
        dim = dim / (16 / sizeof(T));
        repeat_kernel<<<num_tokens, 512, 0, stream.stream>>>(dim, pos, (float4*)input, (float4*)output);
    } else {
        repeat_kernel_2<<<1, num_tokens, 0, stream.stream>>>(pos, input, output);
    }
}

template<typename T>
void log_softmax(const Stream& stream, int32_t num_tokens, int32_t dim, T* input) {
    log_softmax_kernel<<<num_tokens, 1024, 0, stream.stream>>>(dim, input);
}

void init_tree(const Stream& stream, int32_t num_tokens, uint64_t* mask_2d) {
    init_tree_kernel<<<1, num_tokens, 0, stream.stream>>>(mask_2d);
}

void set_parent(const Stream& stream, int32_t num_tokens, int32_t* parent, const int32_t* pos, int32_t offset) {
    set_parent_kernel<<<1, num_tokens, 0, stream.stream>>>(num_tokens, parent, pos, offset);
}

void update_tree(const Stream& stream, int32_t num_tokens, int32_t offset, uint64_t* mask_2d, const uint64_t* tmp_mask_2d, const int32_t* topk_pos) {
    update_tree_kernel<<<1, num_tokens, 0, stream.stream>>>(num_tokens, offset, mask_2d, tmp_mask_2d, topk_pos);
}

template<typename T>
void cumsum(const Stream& stream, int32_t num_tokens, int32_t dim, T* input, const T* weight) {
    cumsum_kernel<<<num_tokens, dim, 0, stream.stream>>>(dim, input, weight);
}

template<typename T>
void remap_hidden(const Stream& stream, int32_t num_tokens, int32_t dim, const int32_t* id_map, const T* real_hidden, T* output, int32_t scale=1) {
    dim = dim / (16 / sizeof(T));
    remap_hidden_kernel<<<num_tokens, 512, 0, stream.stream>>>(scale, dim, id_map, (float4*)real_hidden, (float4*)output);
}

void remap_id(const Stream& stream, int32_t num_tokens, int32_t* id_map, const int32_t* real_id, const int32_t* token_id_remap, int32_t* output=nullptr) {
    if (output == nullptr) output = id_map;
    remap_id_kernel<<<1, num_tokens, 0, stream.stream>>>(id_map, real_id, token_id_remap, output);
}

void make_arange(const Stream& stream, int32_t range, int32_t* offset, int32_t* output) {
    make_arange_kernel<<<1, range, 0, stream.stream>>>(offset, output);
}

__global__ void build_dynamic_tree_kernel(int32_t tree_size, int32_t pos_offset, int32_t topk_per_iter, const int32_t* tried_history_parent, const int32_t* topk_pos, int32_t* tree_pos, uint64_t* tree_mask, int32_t* tree_parent) {
    __shared__ int32_t reverse_tree_id[1024];
    int tid = threadIdx.x;
    if (tid != 0) {
        reverse_tree_id[topk_pos[tid-1]] = tid;
    }
    __syncthreads();
    if (tid == 0) {
        tree_mask[0] = 1;
        tree_pos[0] = pos_offset;
        for (int i = 1; i < tree_size; i++) {
            int p = topk_pos[i-1];
            tree_pos[i] = pos_offset + ((p < topk_per_iter) ?  1 : (p - topk_per_iter) / (topk_per_iter * topk_per_iter) + 2);
            tree_mask[i] = 1ULL << reverse_tree_id[p];;

            if (p < topk_per_iter) {
                p = -1;
            } else {
                p = p - topk_per_iter;
                if (p < topk_per_iter * topk_per_iter) {
                    p = p / topk_per_iter;
                } else {
                    p = tried_history_parent[(p - topk_per_iter * topk_per_iter) / topk_per_iter];
                }
            }
            int parent = p == -1 ? 0 : reverse_tree_id[p];
            tree_parent[i] = parent;
            tree_mask[i] |= tree_mask[parent];
        }
    }
}

void build_dynamic_tree(const Stream& stream, int32_t tree_size, int32_t pos_offset, int32_t topk_per_iter, const int32_t* tried_history_parent, const int32_t* topk_pos, int32_t* tree_pos, uint64_t* tree_mask, int32_t* tree_parent) {
    build_dynamic_tree_kernel<<<1, tree_size, 0, stream.stream>>>(tree_size, pos_offset, topk_per_iter, tried_history_parent, topk_pos, tree_pos, tree_mask, tree_parent);
}


template<typename T>
struct Skip : Norm<T> {
    int dim;

    Skip(int dim) {
        this->dim = dim;
    }

    void init_weight_ptr(Memory* memory) {}

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {}

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output, T* tgt=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        if (prev_output == nullptr) {
            cudaMemcpy(tgt, input, sizeof(T) * this->dim * num_tokens, cudaMemcpyDeviceToDevice);
        } else {
            elementwise_add(stream, num_tokens, this->dim, input, prev_output, tgt);
        }
    }
};

template<typename T>
struct EagleImpl : Model {
    int num_layers;
    int num_iter;
    int topk_per_iter;
    int tree_size;
    int total_tried;

    ModelImpl<T>* model;
    KVCacheManager<T>* kv_caches;
    std::vector<Layer<T>*> layers;
    Linear<T, true, true> *fc1;
    Linear<T> *fc2;
    Linear<T>* lm_head;
    int32_t* token_id_remap;
    functions::TopK<T>* topk_func;
    functions::TopK<T>* topk_func_2;

    T *prev_hidden_state, *prev_embed;
    int num_prev, num_history_tokens;
    int32_t *eagle_position_ids, *eagle_cache_length;
    int *eagle_original_length, eagle_padded_length;
    uint64_t *eagle_mask_2d, *tmp_mask_2d;
    T* eagle_logits;
    T* tired_history_val; int32_t* tired_history_pos;
    int32_t* tired_history_parent;
    bool is_first_draft;
    int V;

    int32_t *h_best, *d_best;    

    T* tmp_kvcache;

    EagleImpl(
        ModelImpl<T>* model,
        int num_layers,
        int num_iter,
        int topk_per_iter,
        int tree_size,
        int V
    ) {
        this->model = model;
        this->num_layers = num_layers;
        this->num_iter = num_iter;
        this->topk_per_iter = topk_per_iter;
        this->tree_size = tree_size;
        this->total_tried = topk_per_iter * topk_per_iter * (num_iter - 1) + topk_per_iter;
        this->V = V;

        kv_caches = new KVCacheManager<T>(num_layers, this->model->num_key_value_heads, this->model->head_dim);
        fc1 = new Linear<T, true, true>(this->model->hidden_size, this->model->hidden_size);
        fc2 = new Linear<T>(this->model->hidden_size, this->model->hidden_size);
        for (int i = 0; i < num_layers; i++) {
            layers.push_back(new Layer<T>(this->model->hidden_size, this->model->intermediate_size, this->model->num_attention_heads, this->model->num_key_value_heads, this->model->head_dim, this->model->rms_norm_eps));
        }
        lm_head = new Linear<T>(this->model->hidden_size, V);

        topk_func = new functions::TopK<T>(V, topk_per_iter);
        topk_func_2 = new functions::TopK<T>(total_tried, this->tree_size-1);
    }

    void init_weight_ptr(Memory* memory) {
        fc1->init_weight_ptr(memory);
        fc2->init_weight_ptr(memory);
        for (int i = 0; i < num_layers; i++) {
            layers[i]->init_weight_ptr(memory);
        }
        lm_head->init_weight_ptr(memory);
        layers[0]->attn->attn_norm = new Skip<T>(this->model->hidden_size);
        kv_caches->rotary_embedding = this->model->kv_caches->rotary_embedding;
        token_id_remap = (int32_t*)memory->allocate_for_model(V * sizeof(int32_t));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        offset = fc1->init_output_ptr(memory, num_tokens, offset);
        offset = fc2->init_output_ptr(memory, num_tokens, offset);
        int64_t layer_end = 0;
        for (int i = 0; i < num_layers; i++) {
            layer_end = layers[i]->init_output_ptr(memory, num_tokens, offset);
        }
        offset = layer_end;
        offset = lm_head->init_output_ptr(memory, 64, offset);
        offset = memory->allocate((void**)&eagle_logits, offset, this->topk_per_iter * V * sizeof(T));
        offset = memory->allocate((void**)&eagle_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tmp_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tired_history_val, offset, this->total_tried * sizeof(T));
        offset = memory->allocate((void**)&tired_history_pos, offset, this->total_tried * sizeof(int32_t));
        offset = memory->allocate((void**)&tired_history_parent, offset, this->topk_per_iter * (this->num_iter - 1) * sizeof(int32_t));
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
            } else if (name.substr(0, 20) == "eagle.token_id_remap") {
                cudaMemcpy((void*)token_id_remap, ptr, V * sizeof(int32_t), cudaMemcpyHostToDevice);
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
            if (name.substr(0, 7) == "lm_head") {
                remap_copy(calc_stream, this->model->lm_head->weight, this->lm_head->weight, this->model->hidden_size, V, this->token_id_remap);
            }
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
            lm_head->prefill(calc_stream, 1, this->fc2->output + (num_prev - 1) * this->model->hidden_size, this->eagle_logits);
            log_softmax(calc_stream, 1, V, this->eagle_logits);
            this->topk_func->prefill(calc_stream, 1, this->eagle_logits);
            cudaMemcpy(this->tired_history_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tired_history_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            remap(calc_stream, topk_per_iter, this->topk_func->topk_pos, this->topk_func_2->topk_pos, this->token_id_remap);
            cudaMemcpy(this->topk_func_2->topk_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
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

            lm_head->prefill(calc_stream, topk_per_iter, this->fc2->output, this->eagle_logits);
            log_softmax(calc_stream, topk_per_iter, V, this->eagle_logits);
            this->topk_func->prefill(calc_stream, topk_per_iter, this->eagle_logits);
            cumsum(calc_stream, topk_per_iter, topk_per_iter, this->topk_func->topk_val, this->topk_func_2->topk_val);
            cudaMemcpy(this->tired_history_val + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_val, topk_per_iter * topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tired_history_pos + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_pos, topk_per_iter * topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->topk_func_2->prefill(calc_stream, 1, this->topk_func->topk_val, topk_per_iter * topk_per_iter, topk_per_iter);

            cudaMemcpy(this->tmp_mask_2d, this->eagle_mask_2d, topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, topk_per_iter, this->tired_history_parent + (d - 1) * topk_per_iter, this->topk_func_2->topk_pos, 10 + (d - 1) * topk_per_iter * topk_per_iter);
            update_tree(calc_stream, topk_per_iter, topk_per_iter * d, this->eagle_mask_2d, this->tmp_mask_2d, this->topk_func_2->topk_pos);
            remap_hidden(calc_stream, topk_per_iter, this->model->hidden_size, this->topk_func_2->topk_pos, this->fc2->output, this->fc1->output, topk_per_iter);
            remap_id(calc_stream, topk_per_iter, this->topk_func_2->topk_pos, this->topk_func->topk_pos, this->token_id_remap);
        }

        this->topk_func_2->prefill(calc_stream, 1, this->tired_history_val);

        // build tree
        build_dynamic_tree(calc_stream, this->tree_size, this->eagle_original_length[0], this->topk_per_iter, this->tired_history_parent, this->topk_func_2->topk_pos, tree_position_ids, tree_attn_mask, tree_parent);
        remap_id(calc_stream, this->tree_size-1, this->topk_func_2->topk_pos, this->tired_history_pos, this->token_id_remap, tree_draft_ids + 1);

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
