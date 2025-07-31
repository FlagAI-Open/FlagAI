#pragma once
#include "../trait.cuh"
#include "norm.cuh"
#include "linear.cuh"
#include "activation.cuh"
#include <cuda_runtime.h>

template <typename T>
struct FFN {
    T* output;
    virtual void init_weight_ptr(Memory* memory) = 0;
    virtual int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) = 0;
    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) = 0;
    virtual void decode(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) = 0;
};

template <typename T>
struct GatedFFN : FFN<T> {
    int hidden_size;
    int intermediate_size;
    float rms_norm_eps;

    Norm<T> *ffn_norm;
    Linear<T> *gate_proj, *up_proj;
    Linear<T> *down_proj;

    T* gated_up;

    GatedFFN(int hidden_size, int intermediate_size, float rms_norm_eps) {
        this->hidden_size = hidden_size;
        this->intermediate_size = intermediate_size;
        this->rms_norm_eps = rms_norm_eps;

        this->ffn_norm = new RMSNorm<T>(hidden_size, rms_norm_eps);
        this->gate_proj = new Linear<T>(hidden_size, intermediate_size);
        this->up_proj = new Linear<T>(hidden_size, intermediate_size);
        this->down_proj = new Linear<T>(intermediate_size, hidden_size);
    }

    void init_weight_ptr(Memory* memory) {
        this->ffn_norm->init_weight_ptr(memory);
        this->gate_proj->init_weight_ptr(memory);
        this->up_proj->init_weight_ptr(memory);
        this->down_proj->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t ffn_norm_end = this->ffn_norm->init_output_ptr(memory, num_tokens, offset);
        int64_t gate_proj_end = this->gate_proj->init_output_ptr(memory, num_tokens, ffn_norm_end);
        int64_t up_proj_end = this->up_proj->init_output_ptr(memory, num_tokens, gate_proj_end);
        int64_t gated_up_end = memory->allocate((void**)&this->gated_up, up_proj_end, num_tokens * intermediate_size * sizeof(T));
        int64_t down_proj_end = this->down_proj->init_output_ptr(memory, num_tokens, gated_up_end);
        this->output = this->down_proj->output;
        return down_proj_end;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("gate_proj") != std::string::npos) {
            this->gate_proj->load_to_storage(name, ptr);
        } else if (name.find("up_proj") != std::string::npos) {
            this->up_proj->load_to_storage(name, ptr);
        } else if (name.find("down_proj") != std::string::npos) {
            this->down_proj->load_to_storage(name, ptr);
        } else if (name.find("post_attention_layernorm") != std::string::npos) {
            this->ffn_norm->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) {
        this->ffn_norm->prefill(stream, num_tokens, input, prev_output);

        // this->gate_proj->prefill(stream, num_tokens, this->ffn_norm->output);
        // this->up_proj->prefill(stream, num_tokens, this->ffn_norm->output);
        // gated_silu<T>(stream, num_tokens, this->intermediate_size, this->gate_proj->output, this->gated_up);
        linear<T>(stream, num_tokens, this->hidden_size, this->intermediate_size*2, this->ffn_norm->output, this->gate_proj->weight, this->gate_proj->output);
        gated_silu_interleaved<T>(stream, num_tokens, this->intermediate_size, this->gate_proj->output, this->gated_up);

        this->down_proj->prefill(stream, num_tokens, this->gated_up);
    }

    void decode(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) {
        prefill(stream, num_tokens, input, prev_output);
    }
};
