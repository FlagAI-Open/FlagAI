#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <stdexcept>

#include "utils.cuh"
#include "trait.cuh"
#include "perf.cuh"
// base model
#include "model/model.cuh"
#include "model/w4a16_gptq_marlin/w4a16_gptq_marlin_model.cuh"
#include "model/minicpm4/minicpm4_model.cuh"
#include "model/minicpm4/minicpm4_w4a16_gptq_marlin_model.cuh"

// eagle
#include "model/eagle.cuh"
#include "model/minicpm4/minicpm4_eagle.cuh"
#include "model/eagle_base_quant/eagle_base_w4a16_gptq_marlin.cuh"

// spec model
#include "model/spec_quant/w4a16_gm_spec_w4a16_gm.cuh"

// hier
#include "model/hier_spec_quant/hier_ea_w4a16_gm_spec_w4a16_gm.cuh"
#include "model/hier_spec_quant/hier_ea_w4a16_gm_rot_spec_w4a16_gm.cuh"


#if defined(ENABLE_DTYPE_FP16) && defined(ENABLE_DTYPE_BF16)
#define DTYPE_SWITCH(COND, ...) \
  [&] { \
    if (COND == 0) { \
      using elem_type = __half; \
      return __VA_ARGS__(); \
    } else { \
      using elem_type = __nv_bfloat16; \
      return __VA_ARGS__(); \
    } \
  }()
#elif defined(ENABLE_DTYPE_FP16)
#define DTYPE_SWITCH(COND, ...) \
  [&] { \
    if (COND != 0) { \
      throw std::runtime_error("BF16 support not compiled. Please recompile with CPMCU_DTYPE=bf16 or CPMCU_DTYPE=fp16,bf16"); \
    } \
    using elem_type = __half; \
    return __VA_ARGS__(); \
  }()
#elif defined(ENABLE_DTYPE_BF16)
#define DTYPE_SWITCH(COND, ...) \
  [&] { \
    if (COND == 0) { \
      throw std::runtime_error("FP16 support not compiled. Please recompile with CPMCU_DTYPE=fp16 or CPMCU_DTYPE=fp16,bf16"); \
    } \
    using elem_type = __nv_bfloat16; \
    return __VA_ARGS__(); \
  }()
#else
#error "At least one of ENABLE_DTYPE_FP16 or ENABLE_DTYPE_BF16 must be defined"
#endif

#define MODEL_TYPE_SWITCH(MODEL_PTR, T, ...)  \
  [&] {                                       \
    if (dynamic_cast<MiniCPM4Impl<T>*>(MODEL_PTR)) {     \
      using ModelType = MiniCPM4Impl<T>; \
      auto* typed_model = static_cast<MiniCPM4Impl<T>*>(MODEL_PTR); \
      return __VA_ARGS__();                   \
    } else if (dynamic_cast<ModelImpl<T>*>(MODEL_PTR)) { \
      using ModelType = ModelImpl<T>; \
      auto* typed_model = static_cast<ModelImpl<T>*>(MODEL_PTR);    \
      return __VA_ARGS__();                   \
    } \
    else if (dynamic_cast<W4A16GPTQMarlinModelImpl<T>*>(MODEL_PTR)) { \
      using ModelType = W4A16GPTQMarlinModelImpl<T>; \
      auto* typed_model = static_cast<W4A16GPTQMarlinModelImpl<T>*>(MODEL_PTR); \
      return __VA_ARGS__();                   \
    } else if (dynamic_cast<MiniCPM4W4A16GPTQMarlinModelImpl<T>*>(MODEL_PTR)) { \
      using ModelType = MiniCPM4W4A16GPTQMarlinModelImpl<T>; \
      auto* typed_model = static_cast<MiniCPM4W4A16GPTQMarlinModelImpl<T>*>(MODEL_PTR); \
      return __VA_ARGS__();                   \
    } \
  }()

#define EAGLE_QUANT_SWITCH(COND, T, ...)               \
  [&] {                                      \
    if (COND == true) {                              \
      using LayerType = W4A16GPTQMarlinLayer<T>; \
      using Fc1Type = W4A16GPTQMarlinLinear<T, true, true>; \
      using Fc2Type = W4A16GPTQMarlinLinear<T>; \
      return __VA_ARGS__();                  \
    } else { \
      using LayerType = Layer<T>; \
      using Fc1Type = Linear<T, true, true>; \
      using Fc2Type = Linear<T>; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

Model* model;

void init_base_model(
    float memory_limit,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int torch_dtype,
    int chunk_length,
    float scale_embed,
    float scale_lmhead,
    float scale_residual
) {
    init_resources();

    DTYPE_SWITCH(torch_dtype, [&] {
        model = new ModelImpl<elem_type>(
            memory_limit,
            vocab_size,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            chunk_length,
            scale_embed,
            scale_lmhead,
            scale_residual
        );
    });

}

void init_minicpm4_model(
    float memory_limit,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int torch_dtype,
    int chunk_length,
    float scale_embed,
    float scale_lmhead,
    float scale_residual,
    int sink_window_size,
    int block_window_size,
    int sparse_topk_k,
    int sparse_switch,
    bool apply_compress_lse
) {
    init_resources();

    DTYPE_SWITCH(torch_dtype, [&] {
        model = new MiniCPM4Impl<elem_type>(
            memory_limit,
            vocab_size,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            chunk_length,
            scale_embed,
            scale_lmhead,
            scale_residual,
            sink_window_size,
            block_window_size,
            sparse_topk_k,
            sparse_switch,
            apply_compress_lse
        );
    });

}

void init_w4a16_gptq_marlin_base_model(
    float memory_limit,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int group_size,
    int torch_dtype,
    int chunk_length,
    float scale_embed,
    float scale_lmhead,
    float scale_residual
) {
    init_resources();

    DTYPE_SWITCH(torch_dtype, [&] {
        model = new W4A16GPTQMarlinModelImpl<elem_type>(
            memory_limit,
            vocab_size,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            group_size,
            chunk_length,
            scale_embed,
            scale_lmhead,
            scale_residual
        );
    });

}

void init_w4a16_gptq_marlin_minicpm4_model(
    float memory_limit,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int group_size,
    int torch_dtype,
    int chunk_length,
    float scale_embed,
    float scale_lmhead,
    float scale_residual,
    int sink_window_size,
    int block_window_size,
    int sparse_topk_k,
    int sparse_switch,
    bool apply_compress_lse
) {
    init_resources();

    DTYPE_SWITCH(torch_dtype, [&] {
        model = new MiniCPM4W4A16GPTQMarlinModelImpl<elem_type>(
            memory_limit,
            vocab_size,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            group_size,
            chunk_length,
            scale_embed,
            scale_lmhead,
            scale_residual,
            sink_window_size,
            block_window_size,
            sparse_topk_k,
            sparse_switch,
            apply_compress_lse
        );
    });

}

// eagle model
void init_eagle_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    bool dispatch_model = false;
    DTYPE_SWITCH(torch_dtype, [&] {
        MODEL_TYPE_SWITCH(model, elem_type, [&] {
            dispatch_model = true;
            model = new EagleImpl<elem_type, ModelType>(
                typed_model,
                num_layers,
                num_iter,
                topk_per_iter,
                tree_size
            );
        });
    });
    if (!dispatch_model) {
        printf("Model type failed to dispatch: %s\n", typeid(*model).name());
    }
}

void init_minicpm4_eagle_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype,
    bool apply_eagle_quant,
    int group_size,
    int eagle_window_size,
    int frspec_vocab_size,
    float residual_scale,
    bool use_input_norm,
    bool use_attn_norm
) {
    bool dispatch_model = false;
    DTYPE_SWITCH(torch_dtype, [&] {
        MODEL_TYPE_SWITCH(model, elem_type, [&] {
            dispatch_model = true;
            EAGLE_QUANT_SWITCH(apply_eagle_quant, elem_type, [&] {
                model = new MiniCPM4EagleImpl<elem_type, ModelType, LayerType, Fc1Type, Fc2Type>(
                    typed_model,
                    num_layers,
                    num_iter,
                    topk_per_iter,
                    tree_size,
                    group_size,
                    eagle_window_size,
                    frspec_vocab_size,
                    residual_scale,
                    use_input_norm,
                    use_attn_norm
                );
            });
        });
    });
    if (!dispatch_model) {
        printf("Model type failed to dispatch: %s\n", typeid(*model).name());
    }
}

// spec model
void init_w4a16_gm_spec_w4a16_gm_model(
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
    bool draft_cuda_graph,
    int torch_dtype
) {
    DTYPE_SWITCH(torch_dtype, [&] {
        model = new W4A16GMSpecW4A16GMImpl<elem_type>(
            (W4A16GPTQMarlinModelImpl<elem_type>*)model,
            draft_vocab_size,
            draft_num_hidden_layers,
            draft_hidden_size,
            draft_intermediate_size,
            draft_num_attention_heads,
            draft_num_key_value_heads,
            draft_head_dim,
            draft_rms_norm_eps,
            draft_group_size,
            num_iter, 
            draft_cuda_graph
        );
    });
}

// hier spec model
void init_hier_eagle_w4a16_gm_spec_w4a16_gm_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int draft_group_size,
    int min_draft_length,
    bool draft_cuda_graph,
    int ea_num_layers,
    int ea_num_iter,
    int ea_topk_per_iter,
    int ea_tree_size,
    bool draft_model_start,
    int torch_dtype
) {
    DTYPE_SWITCH(torch_dtype, [&] {
        model = new HierEagleW4A16GMSpecW4A16GMImpl<elem_type>(
            (W4A16GPTQMarlinModelImpl<elem_type>*)model,
            draft_vocab_size,
            draft_num_hidden_layers,
            draft_hidden_size,
            draft_intermediate_size,
            draft_num_attention_heads,
            draft_num_key_value_heads,
            draft_head_dim,
            draft_rms_norm_eps,
            draft_group_size,
            min_draft_length,
            draft_cuda_graph,
            ea_num_layers,
            ea_num_iter,
            ea_topk_per_iter,
            ea_tree_size,
            draft_model_start
        );
    });
}

void init_hier_eagle_w4a16_gm_rot_spec_w4a16_gm_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int draft_group_size,
    int min_draft_length,
    bool draft_cuda_graph,
    int ea_num_layers,
    int ea_num_iter,
    int ea_topk_per_iter,
    int ea_tree_size,
    bool draft_model_start,
    int torch_dtype
) {
    DTYPE_SWITCH(torch_dtype, [&] {
        model = new HierEagleW4A16GMRotSpecW4A16GMImpl<elem_type>(
            (W4A16GPTQMarlinModelImpl<elem_type>*)model,
            draft_vocab_size,
            draft_num_hidden_layers,
            draft_hidden_size,
            draft_intermediate_size,
            draft_num_attention_heads,
            draft_num_key_value_heads,
            draft_head_dim,
            draft_rms_norm_eps,
            draft_group_size,
            min_draft_length,
            draft_cuda_graph,
            ea_num_layers,
            ea_num_iter,
            ea_topk_per_iter,
            ea_tree_size,
            draft_model_start
        );
    });
}


int init_storage() {
    return model->init_storage();
}

void load_model(std::string name, std::uintptr_t param) {
    model->load_to_storage(name, reinterpret_cast<void*>(param));
}

void prefill(int input_length, int history_length, std::uintptr_t input, std::uintptr_t position_ids, std::uintptr_t output) {
    model->prefill(input_length, history_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), (void*)(output));
}

void decode(int input_length, int padded_length, std::uintptr_t input, std::uintptr_t position_ids, std::uintptr_t cache_length, std::uintptr_t mask_2d, std::uintptr_t output, bool cuda_graph) {
    if (cuda_graph) {
        if (graphCreated_padding_length != padded_length || graphCreated_input_length != input_length) {
            if (graphExec != nullptr) {
                cudaGraphExecDestroy(graphExec);
                graphExec = nullptr;
            }
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
                graph = nullptr;
            }
            cudaStreamBeginCapture(calc_stream.stream, cudaStreamCaptureModeGlobal);
            model->decode(input_length, padded_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(mask_2d), reinterpret_cast<void*>(output));
            cudaStreamEndCapture(calc_stream.stream, &graph);
            cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
            graphCreated_padding_length = padded_length;
            graphCreated_input_length = input_length;
        }
        cudaGraphLaunch(graphExec, calc_stream.stream);
    } else {
        model->decode(input_length, padded_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(mask_2d), reinterpret_cast<void*>(output));
    }
}

void draft(std::uintptr_t tree_draft_ids, std::uintptr_t tree_position_ids, std::uintptr_t cache_length, std::uintptr_t attn_mask, std::uintptr_t tree_parent) {
    model->draft(reinterpret_cast<int32_t*>(tree_draft_ids), reinterpret_cast<int32_t*>(tree_position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(attn_mask), reinterpret_cast<int32_t*>(tree_parent));
}

int verify_and_fix(int num_tokens, std::uintptr_t pred, std::uintptr_t gt, std::uintptr_t position_ids, std::uintptr_t cache_length, std::uintptr_t attn_mask, std::uintptr_t tree_parent) {
    return model->verify(num_tokens, reinterpret_cast<int32_t*>(pred), reinterpret_cast<int32_t*>(gt), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(attn_mask), reinterpret_cast<int32_t*>(tree_parent));
}

void print_perf_summary() {
    perf_summary();
}

PYBIND11_MODULE(C, m) {
    // base bind
    m.def("init_base_model", &init_base_model, "Init base model");
    m.def("init_minicpm4_model", &init_minicpm4_model, "Init minicpm4 model");
    m.def("init_w4a16_gptq_marlin_base_model", &init_w4a16_gptq_marlin_base_model, "Init W4A16 base model");
    m.def("init_w4a16_gptq_marlin_minicpm4_model", &init_w4a16_gptq_marlin_minicpm4_model, "Init W4A16 base model");

    // eagle bind
    m.def("init_eagle_model", &init_eagle_model, "Init eagle model");
    m.def("init_minicpm4_eagle_model", &init_minicpm4_eagle_model, "Init minicpm4 eagle model");
    // spec bind
    m.def("init_w4a16_gm_spec_w4a16_gm_model", &init_w4a16_gm_spec_w4a16_gm_model, "Init w4a16 spec v1 model");

    // hier spec bind
    m.def("init_hier_eagle_w4a16_gm_spec_w4a16_gm_model", &init_hier_eagle_w4a16_gm_spec_w4a16_gm_model, "init hier eagle gm spec gm model");
    m.def("init_hier_eagle_w4a16_gm_rot_spec_w4a16_gm_model", &init_hier_eagle_w4a16_gm_rot_spec_w4a16_gm_model, "init hier eagle rot gm spec gm model");
    
    // interface
    m.def("init_storage", &init_storage, "Init storage");
    m.def("load_model", &load_model, "Load model");
    m.def("prefill", &prefill, "Prefill");
    m.def("decode", &decode, "Decode");
    m.def("draft", &draft, "Draft");
    m.def("verify_and_fix", &verify_and_fix, "Verify and fix");
    m.def("print_perf_summary", &print_perf_summary, "Print perf summary");
} 