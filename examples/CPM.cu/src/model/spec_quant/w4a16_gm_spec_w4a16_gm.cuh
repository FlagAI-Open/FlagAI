#pragma once
#include "../w4a16_gptq_marlin/w4a16_gptq_marlin_model.cuh"
#include "../eagle.cuh"
#include "../drafter.cuh"


template <typename T>
struct W4A16GMSpecW4A16GMImpl: Model {


    W4A16GPTQMarlinModelImpl<T>* draft_model;
    W4A16GPTQMarlinModelImpl<T>* model;

    // draft args
    int32_t *draft_input;
    int32_t *draft_position_ids, *draft_cache_length;
    int * host_draft_cache_length;
    int draft_padded_length; 
    T* draft_logits;
    bool is_first_draft;
    functions::TopK<T>* topk_func;
    int32_t *draft_tmp;
    int32_t *h_best, *d_best;    
    int num_iter;
    int num_prev, num_history_tokens;

    // draft mask always nullptr
    uint64_t* draft_mask_2d;   

    // graph
    bool draft_cuda_graph;
    int draft_graphCreated_padding_length;
    int draft_graphCreated_input_length;
    cudaGraph_t draft_graph;
    cudaGraphExec_t draft_graphExec;

    W4A16GMSpecW4A16GMImpl(
        W4A16GPTQMarlinModelImpl<T>* model,
        int draft_vocab_size,
        int draft_num_hidden_layers,
        int draft_hidden_size,
        int draft_intermediate_size,
        int draft_num_attention_heads,
        int draft_num_key_value_heads,
        int draft_head_dim,
        float draft_rms_norm_eps,
        int draft_group_size,
        int num_iter,
        bool draft_cuda_graph
    ) {
        this->model = model;
        this->draft_model = new W4A16GPTQMarlinModelImpl<T>(
            0,
            draft_vocab_size,
            draft_num_hidden_layers,
            draft_hidden_size,
            draft_intermediate_size,
            draft_num_attention_heads,
            draft_num_key_value_heads,
            draft_head_dim,
            draft_rms_norm_eps,
            draft_group_size,
            this->model->chunk_length
        );

        this->num_iter = num_iter;

        this->draft_mask_2d = 0;
        

        topk_func = new functions::TopK<T>(model->vocab_size, 1); // greedy sample
        
        this->draft_cuda_graph = draft_cuda_graph;
        this->draft_graphCreated_padding_length = -1;
        this->draft_graphCreated_input_length = -1;
        this->draft_graph = nullptr;
        this->draft_graphExec = nullptr;
    }

    

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t lm_head_end = this->draft_model->init_output_ptr(memory, num_tokens, offset);
        offset = lm_head_end;
        
        offset = memory->allocate((void**)&draft_input, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&draft_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&draft_cache_length, offset, sizeof(int32_t));
        cudaMallocHost(&host_draft_cache_length, sizeof(int32_t));

        
        offset = memory->allocate((void**)&draft_logits, offset, 64 * this->draft_model->vocab_size * sizeof(T));
        offset = topk_func->init_output_ptr(memory, 1, offset);
        
        offset = memory->allocate((void**)&draft_tmp, offset, 16*sizeof(int32_t));
        offset = memory->allocate((void**)&d_best, offset, sizeof(int32_t));
        cudaMallocHost(&h_best, sizeof(int32_t));
        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        // this->init_weight_ptr(this->model->memory);
        this->draft_model->init_weight_ptr(this->model->memory);

        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = init_output_ptr(this->model->memory, this->model->chunk_length, offset);

        int model_kv_size = (this->model->num_hidden_layers*this->model->num_key_value_heads*this->model->head_dim);
        int draft_kv_size = (this->draft_model->num_hidden_layers*this->draft_model->num_key_value_heads*this->draft_model->head_dim);
        float ratio = float(model_kv_size)/float(model_kv_size + draft_kv_size);
        kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        this->draft_model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset);
        return min(this->draft_model->kv_caches->budget, this->model->kv_caches->budget);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 5) == "draft"){
            std::string draft_name = name.substr(6);
            this->draft_model->load_to_storage(draft_name, ptr);
        } else {
            this->model->load_to_storage(name, ptr);
        }
    }


    
    void draft_decode_with_graph_control(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        if (this->draft_cuda_graph) {
            if (this->draft_graphCreated_padding_length != padded_length || this->draft_graphCreated_input_length != num_tokens) {
                if (this->draft_graphExec != nullptr) {
                    cudaGraphExecDestroy(this->draft_graphExec);
                    this->draft_graphExec = nullptr;
                }
                if (this->draft_graph != nullptr) {
                    cudaGraphDestroy(this->draft_graph);
                    this->draft_graph = nullptr;
                }
                cudaStreamBeginCapture(calc_stream.stream, cudaStreamCaptureModeGlobal);
                // this->draft_decode(num_tokens, padded_length, output);
                this->draft_model->decode(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output);
                cudaStreamEndCapture(calc_stream.stream, &(this->draft_graph));
                cudaGraphInstantiate(&(this->draft_graphExec), this->draft_graph, nullptr, nullptr, 0);
                this->draft_graphCreated_padding_length = padded_length;
                this->draft_graphCreated_input_length = num_tokens;
            }
            cudaGraphLaunch(this->draft_graphExec, calc_stream.stream);
        } else {
            // this->draft_decode(num_tokens, padded_length, output);
            this->draft_model->decode(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output);
        }
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->prefill(num_tokens, num_history_tokens, input, position_ids, output);
        if (num_history_tokens > 0) {
            this->draft_model->prefill(this->num_prev, this->num_history_tokens, this->draft_input, this->draft_position_ids, this->draft_logits);
        }
        
        cudaMemcpy(this->draft_input, input, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_position_ids, position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->num_prev = num_tokens;
        this->num_history_tokens = num_history_tokens;
        this->is_first_draft = true;
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->decode(num_tokens, padded_length, input, position_ids, cache_length, nullptr, output);
    }

    void draft(int32_t *tree_draft_ids, int32_t *tree_position_ids, int32_t *cache_length, uint64_t*, int32_t*) {
        if (this->is_first_draft) {
            // append tree draft ids to draft input
            cudaMemcpy(this->draft_input+this->num_prev, tree_draft_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_position_ids+this->num_prev, tree_position_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->num_prev += 1;
            this->draft_model->prefill(this->num_prev, this->num_history_tokens, this->draft_input, this->draft_position_ids, (void*)this->draft_logits);

            
            cudaMemcpy(this->draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_cache_length, 1);
            cudaMemcpy(this->draft_position_ids, tree_position_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->host_draft_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
            // this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;
            this->topk_func->prefill(calc_stream, 1, this->draft_logits);
        } else if (this->num_prev == 2){
            // this->draft_decode(this->num_prev, this->draft_padded_length, this->draft_logits);
            this->draft_model->decode(this->num_prev, this->draft_padded_length, this->draft_input, this->draft_position_ids, this->draft_cache_length, nullptr, (void*)this->draft_logits);
            this->topk_func->prefill(calc_stream, 1, this->draft_logits+(this->draft_model->vocab_size));
            add(calc_stream, 1, this->draft_position_ids, 1);
        } else {
            // num_prev == 1
            this->draft_decode_with_graph_control(this->num_prev, this->draft_padded_length, this->draft_input, this->draft_position_ids, this->draft_cache_length, nullptr, (void*)this->draft_logits);
            this->topk_func->prefill(calc_stream, 1, this->draft_logits);
        }
        
        cudaMemcpy(this->draft_input, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_tmp, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);


        for (int d = 1; d < this->num_iter; ++d){
            add(calc_stream, 1, this->draft_cache_length, 1);
            add(calc_stream, 1, this->draft_position_ids, 1);

            this->host_draft_cache_length[0] += 1;
            this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;;
            this->draft_decode_with_graph_control(1, this->draft_padded_length, this->draft_input, this->draft_position_ids, this->draft_cache_length, nullptr, (void*)this->draft_logits);
            this->topk_func->prefill(calc_stream, 1, this->draft_logits);
            cudaMemcpy(this->draft_input, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_tmp + d, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        }

        cudaMemcpy(tree_draft_ids + 1, this->draft_tmp, num_iter*sizeof(int32_t), cudaMemcpyDeviceToDevice);
        make_arange(calc_stream, this->num_iter+1, cache_length, tree_position_ids);
        this->is_first_draft = false;
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* attn_mask, int32_t* tree_parent) { 
        verify_seq_draft(calc_stream, num_tokens, pred, gt, (uint16_t*)attn_mask, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);
        
        if (h_best[0]==(num_iter+1)) {
            // full accept   
            this->num_prev = 2;
            cudaMemcpy(this->draft_input, gt + (num_iter-1), 2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_cache_length, this->h_best[0]+1);
            make_arange(calc_stream, 2, cache_length, this->draft_position_ids);
            add(calc_stream, 2, this->draft_position_ids, num_iter);
            cudaMemcpy(this->host_draft_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
            this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;
        } else {
            this->num_prev = 1;
            cudaMemcpy(this->draft_input, gt + (this->h_best[0]-1), sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_cache_length, this->h_best[0]+1);
            cudaMemcpy(this->draft_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_position_ids, this->h_best[0]);
            cudaMemcpy(this->host_draft_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
            this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;
        }

        return h_best[0];

    }
};