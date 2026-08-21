#pragma once
#include "../attn.cuh"
#include "w4a16_gptq_marlin_linear.cuh"


template <typename T>
struct W4A16GPTQMarlinAttention {
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    float rms_norm_eps;

    Norm<T> *attn_norm;
    W4A16GPTQMarlinLinear<T> *qkv_proj;
    W4A16GPTQMarlinLinear<T> *o_proj;
    T* output;

    T* attn_output;
    float *softmax_lse, *softmax_lse_accum, *oaccum;

    int window_size;

    T* q_proj_output, *v_proj_output, *k_proj_output; 
    T* permute_qkv_output;

    W4A16GPTQMarlinAttention(int hidden_size, int num_attention_heads, int num_key_value_heads, int head_dim, float rms_norm_eps, int group_size, int window_size = 0) {
        this->hidden_size = hidden_size;
        this->num_attention_heads = num_attention_heads;
        this->num_key_value_heads = num_key_value_heads;
        this->head_dim = head_dim;
        this->rms_norm_eps = rms_norm_eps;

        this->attn_norm = new RMSNorm<T>(hidden_size, rms_norm_eps);

        this->qkv_proj = new W4A16GPTQMarlinLinear<T>(hidden_size, (num_attention_heads + 2*num_key_value_heads) * head_dim, group_size);
        
        this->o_proj = new W4A16GPTQMarlinLinear<T>(hidden_size, num_attention_heads * head_dim, group_size);

        this->window_size = window_size;
    }

    void init_weight_ptr(Memory* memory) {
        this->attn_norm->init_weight_ptr(memory);
        this->qkv_proj->init_weight_ptr(memory);
        this->o_proj->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t attn_norm_end = this->attn_norm->init_output_ptr(memory, num_tokens, offset);
        int64_t qkv_proj_end = this->qkv_proj->init_output_ptr(memory, num_tokens, attn_norm_end);

        this->q_proj_output = this->qkv_proj->output;
        this->k_proj_output = this->qkv_proj->output + num_tokens * this->num_attention_heads * this->head_dim;
        this->v_proj_output = this->qkv_proj->output + num_tokens * (this->num_attention_heads+this->num_key_value_heads) * this->head_dim;
        int64_t qkv_permute_end = memory->allocate((void**)&this->permute_qkv_output, qkv_proj_end, num_tokens * (this->num_attention_heads + 2*this->num_key_value_heads) * this->head_dim * sizeof(T));
        
        int64_t attn_output_end = memory->allocate((void**)&this->attn_output, offset, num_tokens * this->num_attention_heads * this->head_dim * sizeof(T));
        int64_t softmax_lse_end = memory->allocate((void**)&this->softmax_lse, qkv_permute_end, num_tokens * this->num_attention_heads * sizeof(float));
        const int max_num_splits = 128; // Maximum number of splits for attention computation
        const int max_spec_tree_size = 64; // Maximum size of speculative decoding tree
        int64_t softmax_lse_accum_end = memory->allocate((void**)&this->softmax_lse_accum, softmax_lse_end, max(max_num_splits * max_spec_tree_size, num_tokens) * this->num_attention_heads * sizeof(float));
        int64_t oaccum_end = memory->allocate((void**)&this->oaccum, softmax_lse_accum_end, max(max_num_splits * max_spec_tree_size, num_tokens) * this->num_attention_heads * this->head_dim * sizeof(float));

        int64_t o_proj_end = this->o_proj->init_output_ptr(memory, num_tokens, qkv_permute_end);
        this->output = this->o_proj->output;

        return std::max(oaccum_end, o_proj_end);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("qkv_proj") != std::string::npos) {
            this->qkv_proj->load_to_storage(name, ptr);
        } else if (name.find("o_proj") != std::string::npos) {
            this->o_proj->load_to_storage(name, ptr);
        } else if (name.find("input_layernorm") != std::string::npos) {
            this->attn_norm->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Attn Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, int32_t num_history_tokens, T* input, T* prev_output, int32_t* position_ids, KVCache<T>* kv_cache, T* a_tmp, float* c_tmp) {
        T* k_cache = kv_cache->offset_k(num_history_tokens);
        T* v_cache = kv_cache->offset_v(num_history_tokens);

        this->attn_norm->prefill(stream, num_tokens, input, prev_output);
        this->qkv_proj->prefill(stream, num_tokens, this->attn_norm->output, a_tmp, c_tmp);
        permute(stream, num_tokens, this->num_attention_heads * this->head_dim, this->num_key_value_heads * this->head_dim, this->qkv_proj->output, this->permute_qkv_output);
        cudaMemcpy(k_cache, this->permute_qkv_output + num_tokens*this->num_attention_heads*this->head_dim, num_tokens*this->num_key_value_heads*this->head_dim*sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(v_cache, this->permute_qkv_output + num_tokens*( this->num_attention_heads + this->num_key_value_heads)*this->head_dim, num_tokens*this->num_key_value_heads*this->head_dim*sizeof(T), cudaMemcpyDeviceToDevice);
        

        kv_cache->rotary_embedding->prefill(stream, num_tokens, this->num_attention_heads, this->num_key_value_heads, this->permute_qkv_output, k_cache, position_ids);

        cuda_perf_start_on_stream_f(Q_PREFILL_ATTN_CORE, stream.stream);
        mha_fwd_kvcache(
            TypeTraits<T>::type_code()==1,
            1,
            num_tokens,
            num_history_tokens+num_tokens,
            this->num_attention_heads,
            this->num_key_value_heads,
            this->head_dim,
            this->permute_qkv_output,
            kv_cache->k_cache,
            kv_cache->v_cache,
            nullptr,
            Mask(nullptr),
            this->attn_output,
            this->softmax_lse,
            this->softmax_lse_accum,
            this->oaccum,
            rsqrtf(float(this->head_dim)),
            true,
            -1,
            -1,
            0,
            stream.stream,
            nullptr,
            this->window_size
        );
        cuda_perf_stop_on_stream_f(Q_PREFILL_ATTN_CORE, stream.stream);

        // flash attention and put output to attn_norm->output
        this->o_proj->prefill(stream, num_tokens, this->attn_output, a_tmp, c_tmp);
    }

    void decode(const Stream& stream, int32_t num_tokens, int32_t padded_length, T* input, T* prev_output, int32_t* position_ids, int32_t* cache_length, const Mask& mask, KVCache<T>* kv_cache, T* a_tmp, float* c_tmp) {
        this->attn_norm->prefill(stream, num_tokens, input, prev_output);
        T *q, *k, *v;
        
        if (num_tokens > 1) {
            this->qkv_proj->prefill(stream, num_tokens, this->attn_norm->output, a_tmp, c_tmp);
            permute(stream, num_tokens, this->num_attention_heads * this->head_dim, this->num_key_value_heads * this->head_dim, this->qkv_proj->output, this->permute_qkv_output); // TODO: Double check
            q = this->permute_qkv_output;
        } else {
            this->qkv_proj->prefill(stream, num_tokens, this->attn_norm->output, a_tmp, c_tmp);
            q = this->qkv_proj->output;
        }
        k = q + num_tokens * this->num_attention_heads * this->head_dim;
        v = k + num_tokens * this->num_key_value_heads * this->head_dim;
        kv_cache->rotary_embedding->prefill(stream, num_tokens, this->num_attention_heads, this->num_key_value_heads, q, k, position_ids);

        copy_to_kvcache(stream, num_tokens, k, v, kv_cache, cache_length);

        cuda_perf_start_on_stream_f(Q_DECODE_ATTN_CORE, stream.stream);
        mha_fwd_kvcache(
            TypeTraits<T>::type_code()==1,
            1,
            num_tokens,
            padded_length,
            this->num_attention_heads,
            this->num_key_value_heads,
            this->head_dim,
            q,
            kv_cache->k_cache,
            kv_cache->v_cache,
            cache_length,
            mask,
            this->attn_output,
            this->softmax_lse,
            this->softmax_lse_accum,
            this->oaccum,
            rsqrtf(float(this->head_dim)),
            true,
            -1,
            -1,
            0,
            stream.stream,
            nullptr,
            this->window_size
        );
        cuda_perf_stop_on_stream_f(Q_DECODE_ATTN_CORE, stream.stream);
        // flash attention and put output to attn_norm->output
        this->o_proj->prefill(stream, num_tokens, this->attn_output, a_tmp, c_tmp);
    }
};