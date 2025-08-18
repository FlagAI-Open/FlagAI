#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "../linear.cuh"
#include "../../qgemm/gptq_marlin/marlin.cuh"
#include "../../qgemm/gptq_marlin/gptq_marlin.cuh"
#include "../../qgemm/gptq_marlin/core/scalar_type.hpp"

template <typename T, bool transposed=true, bool has_bias=false>
struct W4A16GPTQMarlinLinear {
    int dim_in;
    int dim_out;

    T* output;
    int32_t* weight;
    T* bias;
    T* scales;

    // just placeholder
    int32_t* qzeros;
    int32_t* g_idx;
    int32_t* perm;
    int32_t* workspace;

    // new added
    const vllm::ScalarType weight_scalar_dtype;
    bool is_k_full;
    bool use_fp32_reduce; // be true is better
    
    int num_groups;
    int group_size;
    bool has_act_order;


    W4A16GPTQMarlinLinear(int dim_in, int dim_out, int group_size)
                    :weight_scalar_dtype(static_cast<uint8_t>(0), 
                              static_cast<uint8_t>(4), 
                              false, 
                              static_cast<int32_t>(8), 
                              false) // Initialize weight_scalar_dtype in the constructor 
    {
        this->dim_in = dim_in;
        this->dim_out = dim_out;

        // place holder
        this->qzeros = 0;
        this->g_idx = 0;
        this->perm = 0;

        this->is_k_full = true;
        this->use_fp32_reduce = true;
        this->group_size = group_size;
        if (this->group_size == 128){
            this->num_groups = (dim_in) / group_size;
        } else if (this->group_size == -1){
            this->num_groups = 1;
        } else {
            throw std::invalid_argument("Unsupported group size");
        }
            
        this->has_act_order = false;
        
    }

    void init_weight_ptr(Memory* memory) {
        const int w_size = this->dim_in * this->dim_out / 8;
        weight = (int32_t*)memory->allocate_for_model(w_size*sizeof(int32_t));
        const int s_size = this->num_groups * this->dim_out ;
        scales = (T*)memory->allocate_for_model(s_size * sizeof(T));

        const int workspace_size = (this->dim_out / 64)*16;
        workspace = (int32_t*)memory->allocate_for_model(workspace_size * sizeof(int32_t));

        if constexpr (has_bias) {
            bias = (T*)memory->allocate_for_model(dim_out * sizeof(T));
        }
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t output_offset = memory->allocate((void**)&this->output, offset, num_tokens * dim_out * sizeof(T));
        return output_offset;
    }
    

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("scales") != std::string::npos) {
            const int s_size = this->num_groups * this->dim_out;
            cudaMemcpy((void*)scales, ptr, s_size*sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find("qweight") != std::string::npos) {
            const int w_size = this->dim_in * this->dim_out / 8;
            cudaMemcpy((void*)weight, ptr, w_size*sizeof(int32_t), cudaMemcpyHostToDevice);
        } else if (name.find("bias") != std::string::npos) {
            cudaMemcpy((void*)bias, ptr, dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else {
            throw std::invalid_argument("Linear Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* a_tmp, float* c_tmp, T* tgt=nullptr, bool inplace=false) {
        T* tgt_temp;
        if (tgt == nullptr) {
            tgt_temp = this->output;
            tgt = tgt_temp;
        } else if (inplace && tgt) {
            tgt_temp = this->output;
        }
        else if (!inplace && tgt) {
            tgt_temp = tgt;
        }
        gptq_marlin_gemm<T>(
            input,
            weight,
            scales,
            qzeros,
            g_idx,
            perm,
            workspace,
            weight_scalar_dtype,
            num_tokens,
            dim_out,
            dim_in,
            is_k_full,
            false,
            use_fp32_reduce,
            tgt_temp,
            num_groups,
            group_size,
            2*dim_out,
            has_act_order,
            stream.stream,
            a_tmp,
            c_tmp
        );

        if (inplace) {
            elementwise_add<T>(stream, num_tokens, this->dim_out, tgt, tgt_temp, tgt);
        }
        if constexpr (has_bias) {
            batched_add<T>(stream, num_tokens, this->dim_out, tgt, this->bias, tgt);
        }
    }
};