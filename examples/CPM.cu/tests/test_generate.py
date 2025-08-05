import torch
from cpmcu.llm import LLM
from cpmcu.llm_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
from cpmcu.speculative import LLM_with_eagle
from cpmcu.speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
from transformers import AutoTokenizer
import time
import numpy as np
import argparse
import sys
import os
from huggingface_hub import snapshot_download   

# Default Configuration
default_config = {
    'test_minicpm4': True,
    'use_stream': True,
    'apply_eagle': True,
    'apply_quant': True,
    'apply_sparse': True,
    'apply_eagle_quant': True,
    "minicpm4_yarn": True, # TODO default is True for long context test, better implementation
    'frspec_vocab_size': 32768,
    'eagle_window_size': 8 * 128,
    'eagle_num_iter': 2,
    'eagle_topk_per_iter': 10,
    'eagle_tree_size': 12,
    'apply_compress_lse': True,
    'sink_window_size': 1,
    'block_window_size': 8,
    'sparse_topk_k': 64,
    "sparse_switch": 1,
    'num_generate': 256,
    'chunk_length': 2048,
    'memory_limit': 0.9,
    'cuda_graph': True,
    'dtype': torch.float16,
    'use_terminators': True,
    "temperature": 0.0,
    "random_seed": None,
}

# Demo Configuration: Only for MiniCPM4 demo, will be deleted after release
demo_config = {
    'use_enter': False,
    'use_decode_enter': False,
}

# Combined Default Configurations
default_config = {**default_config, **demo_config}

def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(description='Generate text using LLM models')
    
    # Basic arguments
    parser.add_argument('--path-prefix', '--path_prefix', '-p', type=str, default='openbmb', 
                        help='Path prefix for model directories, you can use openbmb to download models, or your own path (default: openbmb)')

    # Prompt arguments
    parser.add_argument('--prompt-file', '--prompt_file', type=str, default=None,
                        help='Path to prompt file (default: None)')
    parser.add_argument('--prompt-text', '--prompt_text', type=str, default=None,
                        help='Direct prompt text (default: None)')
    parser.add_argument('--prompt-haystack', '--prompt_haystack', type=int, default=15,
                        help='Generate haystack prompt with specified length in thousands (e.g., 120 for 120k tokens)')

    # Model configuration boolean arguments
    parser.add_argument('--test-minicpm4', '--test_minicpm4', action='store_true',
                        help='Use MiniCPM4 model')
    parser.add_argument('--no-test-minicpm4', '--no_test_minicpm4', action='store_false', dest='test_minicpm4',
                        help='Do not use MiniCPM4 model')
    parser.add_argument('--use-stream', '--use_stream', action='store_true',
                        help='Use stream generation')
    parser.add_argument('--no-use-stream', '--no_use_stream', action='store_false', dest='use_stream',
                        help='Do not use stream generation')
    parser.add_argument('--apply-eagle', '--apply_eagle', action='store_true',
                        help='Use Eagle speculative decoding')
    parser.add_argument('--no-apply-eagle', '--no_apply_eagle', action='store_false', dest='apply_eagle',
                        help='Do not use Eagle speculative decoding')
    parser.add_argument('--apply-quant', '--apply_quant', action='store_true',
                        help='Use quantized model')
    parser.add_argument('--no-apply-quant', '--no_apply_quant', action='store_false', dest='apply_quant',
                        help='Do not use quantized model')
    parser.add_argument('--apply-sparse', '--apply_sparse', action='store_true',
                        help='Use sparse attention')
    parser.add_argument('--no-apply-sparse', '--no_apply_sparse', action='store_false', dest='apply_sparse',
                        help='Do not use sparse attention')
    parser.add_argument('--apply-eagle-quant', '--apply_eagle_quant', action='store_true',
                        help='Use quantized Eagle model')
    parser.add_argument('--no-apply-eagle-quant', '--no_apply_eagle_quant', action='store_false', dest='apply_eagle_quant',
                        help='Do not use quantized Eagle model')
    parser.add_argument('--apply-compress-lse', '--apply_compress_lse', action='store_true',
                        help='Apply LSE compression, only support on sparse attention, this will compress the stage 1 kv twice for LSE pre-computing')
    parser.add_argument('--no-apply-compress-lse', '--no_apply_compress_lse', action='store_false', dest='apply_compress_lse',
                        help='Do not apply LSE compression')
    parser.add_argument('--cuda-graph', '--cuda_graph', action='store_true',
                        help='Use CUDA graph optimization')
    parser.add_argument('--no-cuda-graph', '--no_cuda_graph', action='store_false', dest='cuda_graph',
                        help='Do not use CUDA graph optimization')
    parser.add_argument('--use-teminators', '--use_terminators', action='store_true',
                        help='Use teminators, if not specified, the generation will not be interrupted')
    parser.add_argument('--no-use-teminators', '--no_use_terminators', action='store_false', dest='use_terminators',
                        help='Do not use teminators')
    parser.add_argument('--minicpm4-yarn', '--minicpm4_yarn', action='store_true',
                        help='Use MiniCPM4 YARN, this is for very long context, such as > 32/64k tokens')
    parser.add_argument('--no-minicpm4-yarn', '--no_minicpm4_yarn', action='store_false', dest='minicpm4_yarn',
                        help='Do not use MiniCPM4 YARN')

    # Model configuration numeric arguments
    parser.add_argument('--frspec-vocab-size', '--frspec_vocab_size', type=int, default=None,
                        help='Frequent speculation vocab size (default: from default_config)')
    parser.add_argument('--eagle-window-size', '--eagle_window_size', type=int, default=None,
                        help='Eagle window size (default: from default_config)')
    parser.add_argument('--eagle-num-iter', '--eagle_num_iter', type=int, default=None,
                        help='Eagle number of iterations (default: from default_config)')
    parser.add_argument('--eagle-topk-per-iter', '--eagle_topk_per_iter', type=int, default=None,
                        help='Eagle top-k per iteration (default: from default_config)')
    parser.add_argument('--eagle-tree-size', '--eagle_tree_size', type=int, default=None,
                        help='Eagle tree size (default: from default_config)')
    parser.add_argument('--sink-window-size', '--sink_window_size', type=int, default=None,
                        help='Sink window size of sparse attention (default: from default_config)')
    parser.add_argument('--block-window-size', '--block_window_size', type=int, default=None,
                        help='Block window size of sparse attention (default: from default_config)')
    parser.add_argument('--sparse-topk-k', '--sparse_topk_k', type=int, default=None,
                        help='Sparse attention top-k (default: from default_config)')
    parser.add_argument('--sparse-switch', '--sparse_switch', type=int, default=None,
                        help='Context length of dense and sparse attention switch (default: from default_config)')
    parser.add_argument('--num-generate', '--num_generate', type=int, default=None,
                        help='Number of tokens to generate (default: from default_config)')
    parser.add_argument('--chunk-length', '--chunk_length', type=int, default=None,
                        help='Chunk length for prefilling (default: from default_config)')
    parser.add_argument('--memory-limit', '--memory_limit', type=float, default=None,
                        help='Memory limit for use (default: from default_config)')
    parser.add_argument('--temperature', '--temperature', type=float, default=None,
                        help='Temperature for processing (default: from default_config)')
    parser.add_argument('--dtype', type=str, default=None, choices=['float16', 'bfloat16'],
                        help='Model dtype (default: from default_config)')
    parser.add_argument('--random-seed', '--random_seed', type=int, default=None,
                        help='Random seed for processing (default: from default_config)')
    # Demo arguments
    parser.add_argument('--use-enter', '--use_enter', action='store_true',
                        help='Use enter to generate')
    parser.add_argument('--no-use-enter', '--no_use_enter', action='store_false', dest='use_enter',
                        help='Do not use enter to generate')
    parser.add_argument('--use-decode-enter', '--use_decode_enter', action='store_true',
                        help='Use enter before decode phase')
    parser.add_argument('--no-use-decode-enter', '--no_use_decode_enter', action='store_false', dest='use_decode_enter',
                        help='Do not use enter before decode phase')
    
    return parser

def parse_and_merge_config(default_config):
    """Parse arguments and merge with default configuration"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set default values to None for boolean arguments that weren't specified
    bool_args = [key for key, value in default_config.items() if isinstance(value, bool)]
    for arg in bool_args:
        # Convert underscores to hyphens for command line argument names
        arg_hyphen = arg.replace('_', '-')
        # Check for both formats (hyphen and underscore)
        arg_specified = (f'--{arg_hyphen}' in sys.argv or f'--no-{arg_hyphen}' in sys.argv or
                        f'--{arg}' in sys.argv or f'--no-{arg}' in sys.argv)
        if not arg_specified:
            setattr(args, arg, None)

    # Override default config with command line arguments
    config = default_config.copy()

    # Define parameter mappings for automatic override (exclude dtype which needs special handling)
    auto_override_params = [key for key in default_config.keys() if key != 'dtype']

    # Override config values if arguments are provided
    for param in auto_override_params:
        arg_value = getattr(args, param)
        if arg_value is not None:
            config[param] = arg_value

    # Handle dtype separately due to type conversion
    if args.dtype is not None:
        config['dtype'] = torch.float16 if args.dtype == 'float16' else torch.bfloat16
    
    return args, config

def check_or_download_model(path):
    if os.path.exists(path):
        return path
    else:
        cache_dir = snapshot_download(path)
        return cache_dir

def get_model_paths(path_prefix, config):
    """Get model paths based on configuration"""
    if config['test_minicpm4']:
        if config['apply_eagle_quant']:
            eagle_repo_id = f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu"
        else:
            eagle_repo_id = f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec"
    else:
        eagle_repo_id = f"{path_prefix}/EAGLE-LLaMA3-Instruct-8B"
    
    if not config['apply_quant']:
        if config['test_minicpm4']:
            base_repo_id = f"{path_prefix}/MiniCPM4-8B"
        else:
            base_repo_id = f"{path_prefix}/Meta-Llama-3-8B-Instruct"
    else:
        base_repo_id = f"{path_prefix}/MiniCPM4-8B-marlin-cpmcu"

    eagle_path = check_or_download_model(eagle_repo_id)
    base_path = check_or_download_model(base_repo_id)
    
    return eagle_path, base_path, eagle_repo_id, base_repo_id

def apply_minicpm4_yarn_config(llm, config):
    """Apply MiniCPM4 YARN configuration to model config"""
    yarn_factors = [
        0.9977997200264581, 1.014658295992452, 1.0349680404997148, 1.059429246056193,
        1.0888815016813513, 1.1243301355211495, 1.166977103606075, 1.2182568066927284,
        1.2798772354275727, 1.3538666751582975, 1.4426259039919596, 1.5489853358570191,
        1.6762658237220625, 1.8283407612492941, 2.0096956085876183, 2.225478927469756,
        2.481536379650452, 2.784415934557119, 3.1413289096347365, 3.560047844772632,
        4.048719380066383, 4.752651957515948, 5.590913044973868, 6.584005926629993,
        7.7532214876576155, 9.119754865903639, 10.704443927019176, 12.524994176518703,
        14.59739595363613, 16.93214476166354, 19.53823297353041, 22.417131025031697,
        25.568260840911098, 28.991144156566317, 32.68408069090375, 36.65174474170465,
        40.90396065611201, 45.4664008671033, 50.37147343433591, 55.6804490772103,
        61.470816952306556, 67.8622707390618, 75.00516023410414, 83.11898235973767,
        92.50044360202462, 103.57086856690864, 116.9492274587385, 118.16074567836519,
        119.18497548708795, 120.04810876261652, 120.77352815196981, 121.38182790207875,
        121.89094985353891, 122.31638758099915, 122.6714244963338, 122.9673822552567,
        123.21386397019609, 123.41898278254268, 123.58957065488238, 123.73136519024158,
        123.84917421274221, 123.94701903496814, 124.02825801299717, 124.09569231686116
    ]
    
    # Create or modify rope_scaling configuration
    if not hasattr(llm.config, 'rope_scaling') or llm.config.rope_scaling is None:
        llm.config.rope_scaling = {}
        
    llm.config.rope_scaling['rope_type'] = 'longrope'
    llm.config.rope_scaling['long_factor'] = yarn_factors
    llm.config.rope_scaling['short_factor'] = yarn_factors
    print("Forcing MiniCPM4 YARN rope_scaling parameters")

def create_model(eagle_path, base_path, config):
    """Create model instance based on configuration"""
    common_kwargs = {
        'dtype': config['dtype'],
        'chunk_length': config['chunk_length'],
        'cuda_graph': config['cuda_graph'],
        'apply_sparse': config['apply_sparse'],
        'sink_window_size': config['sink_window_size'],
        'block_window_size': config['block_window_size'],
        'sparse_topk_k': config['sparse_topk_k'],
        'sparse_switch': config['sparse_switch'],
        'apply_compress_lse': config['apply_compress_lse'],
        'memory_limit': config['memory_limit'],
        'use_enter': config['use_enter'],
        'use_decode_enter': config['use_decode_enter'],
        'temperature': config['temperature'],
        'random_seed': config['random_seed'],
    }
    
    eagle_kwargs = {
        'num_iter': config['eagle_num_iter'],
        'topk_per_iter': config['eagle_topk_per_iter'],
        'tree_size': config['eagle_tree_size'],
        'eagle_window_size': config['eagle_window_size'],
        'frspec_vocab_size': config['frspec_vocab_size'],
        'apply_eagle_quant': config['apply_eagle_quant'],
        'use_rope': config['test_minicpm4'],
        'use_input_norm': config['test_minicpm4'],
        'use_attn_norm': config['test_minicpm4']
    }
    
    if config['apply_quant']:
        if config['apply_eagle']:
            return W4A16GPTQMarlinLLM_with_eagle(eagle_path, base_path, **common_kwargs, **eagle_kwargs)
        else:
            return W4A16GPTQMarlinLLM(base_path, **common_kwargs)
    else:
        if config['apply_eagle']:
            return LLM_with_eagle(eagle_path, base_path, **common_kwargs, **eagle_kwargs)
        else:
            return LLM(base_path, **common_kwargs)

def make_input(tokenizer, args, prompt_content=None):
    """Prepare input tokens from prompt content or file"""
    
    def make_haystack_prompt(digits, target_length_k):
        """Generate haystack prompt with pass key hidden in context"""
        # Simple calculation based on target length
        a = target_length_k * 16  # Scale factor for before text
        b = target_length_k * 33  # Scale factor for after text
        
        head = "There is a pass key hidden in the context. Find it and remember it. I will quiz you about it later. "
        before = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * a
        needle = f"The pass key is {digits}. Remember it. The pass key is {digits}"
        after = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * b
        query = "Now, give me the exact number of the pass key. The pass key is "
        return head + before + needle + after + query
    
    if prompt_content is None:
        # Check if file or text was specified
        file_specified = args.prompt_file is not None
        text_specified = args.prompt_text is not None
        
        if not file_specified and not text_specified:
            # Case 1: Neither file nor text specified, use haystack with default value
            print(f"Using haystack prompt with {args.prompt_haystack}k tokens (default)")
            prompt_content = make_haystack_prompt(681725493, args.prompt_haystack)
        else:
            # Case 2 & 3: At least one of file or text specified, ignore haystack
            prompt_content = ""
            
            # Load from file if specified
            if file_specified:
                try:
                    with open(args.prompt_file, 'r', encoding='utf-8') as f:
                        file_content = f.read().strip()
                    prompt_content += file_content
                    print(f"Loaded prompt from file: {args.prompt_file}")
                except FileNotFoundError:
                    print(f"Warning: {args.prompt_file} not found, skipping file content")
                except Exception as e:
                    print(f"Error reading {args.prompt_file}: {e}, skipping file content")
            
            # Append text if specified
            if text_specified:
                if file_specified and prompt_content:
                    # Case 3: Both specified, append text to file content
                    prompt_content += "\n" + args.prompt_text
                    print(f"Appended prompt text to file content")
                else:
                    # Case 2: Only text specified
                    prompt_content = args.prompt_text
                    print(f"Using direct prompt text input")
            
            # Fallback if no content was loaded
            if not prompt_content:
                print(f"No valid content found, using default Chinese prompt")
                prompt_content = "北京有哪些好玩的地方"
    
    if config['test_minicpm4'] and not file_specified and not text_specified: # TODO: haystack need w/o chat template, may be a bug
        prompt = prompt_content
    else:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_content}], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
    
    print(f"Input token count: {input_ids.shape[1]}")
    if input_ids.shape[1] <= 100:  # Only show input_ids for short prompts
        print(f"Input_ids: {input_ids}")
    
    return input_ids

def print_generation_summary(mode, prefill_stats, decode_stats, config):
    """Print unified generation summary for both modes"""
    print("\n" + "=" * 50)
    print(f"{mode} Generation Summary:")
    print("=" * 50)
    
    # Prefill statistics
    print(f"Prefill length: {prefill_stats['length']}")
    print(f"Prefill time: {prefill_stats['time']:.2f} s")
    print(f"Prefill tokens/s: {prefill_stats['tokens_per_sec']:.2f}")

    # Eagle-specific statistics
    if config['apply_eagle'] and 'mean_accept_length' in decode_stats:
        print(f"Mean accept length: {decode_stats['mean_accept_length']:.2f}")
        # print(f"Decode token/s when acc = 1: {decode_stats['tokens_per_sec'] / decode_stats['mean_accept_length']:.2f}")
    
    # Decode statistics
    print(f"Decode length: {decode_stats['length']}")
    print(f"Decode time: {decode_stats['time']:.2f} s")
    print(f"Decode tokens/s: {decode_stats['tokens_per_sec']:.2f}")

def run_stream_generation(llm, input_ids, config, teminators, tokenizer):
    """Run streaming generation and display results"""
    print("\nGenerated text (streaming output):")
    print("-" * 50)
    
    # Statistics tracking
    prefill_length = input_ids.shape[1]
    prefill_time = 0.0
    total_decode_time = 0.0
    
    generated_text = ""
    total_decode_tokens = 0
    accept_lengths = []
    
    try:
        for result in llm.generate(input_ids, config['num_generate'], teminators=teminators, use_stream=True):
            token = result['token']
            text = result['text']
            is_finished = result['is_finished']
            
            # Track timing statistics
            if 'prefill_time' in result and result['prefill_time'] > 0:
                prefill_time = result['prefill_time']
            if 'decode_time' in result and result['decode_time'] > 0:
                total_decode_time = result['decode_time']
            
            generated_text += text
            total_decode_tokens += 1
            
            # Track accept lengths for eagle models
            if 'accept_length' in result and result['accept_length'] > 0:
                accept_lengths.append(result['accept_length'])
            
            print(text, end='', flush=True)
            
            if is_finished:
                break
                
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
    
    prefill_stats = {
        'length': prefill_length,
        'time': prefill_time,
        'tokens_per_sec': prefill_length / prefill_time if prefill_time > 0 else 0
    }
    
    decode_stats = {
        'length': total_decode_tokens,
        'time': total_decode_time,
        'tokens_per_sec': total_decode_tokens / total_decode_time if total_decode_time > 0 else 0
    }
    
    if config['apply_eagle'] and accept_lengths:
        decode_stats['mean_accept_length'] = np.mean(accept_lengths)
    
    print_generation_summary("Stream", prefill_stats, decode_stats, config)

def run_non_stream_generation(llm, input_ids, config, teminators, tokenizer):
    """Run non-stream generation and display results"""
    prefill_length = input_ids.shape[1]
    
    torch.cuda.synchronize()
    start_time = time.time()
    gen_result = llm.generate(input_ids, config['num_generate'], teminators=teminators, use_stream=False)
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Handle different return formats based on model type
    if config['apply_eagle']:
        # Eagle models return: (tokens, accept_lengths, decode_time, prefill_time)
        tokens, accept_lengths, decode_time, prefill_time = gen_result
        decode_length = len(tokens)
        mean_accept_length = np.mean(accept_lengths)
    else:
        # Base models return: (tokens, decode_time, prefill_time)
        tokens, decode_time, prefill_time = gen_result
        decode_length = len(tokens)
        mean_accept_length = None

    print("\n[Generated Result]")
    print(tokenizer.decode(tokens).strip())
    print("\n")

    prefill_stats = {
        'length': prefill_length,
        'time': prefill_time,
        'tokens_per_sec': prefill_length / prefill_time if prefill_time > 0 else 0
    }
    
    decode_stats = {
        'length': decode_length,
        'time': decode_time,
        'tokens_per_sec': decode_length / decode_time if decode_time > 0 else 0
    }
    
    if mean_accept_length is not None:
        decode_stats['mean_accept_length'] = mean_accept_length
    
    print_generation_summary("Non-Stream", prefill_stats, decode_stats, config)

def print_config(config, use_stream):
    """Print all configuration parameters"""
    print("=" * 50)
    print("Configuration Parameters:")
    print("=" * 50)
    print(f"Features: eagle={config['apply_eagle']}, quant={config['apply_quant']}, sparse={config['apply_sparse']}")
    print(f"Generation: num_generate={config['num_generate']}, chunk_length={config['chunk_length']}, use_terminators={config['use_terminators']}, use_stream={config['use_stream']}")
    print(f"Sampling: temperature={config['temperature']}, random_seed={config['random_seed']}")
    print(f"Demo: use_enter={config['use_enter']}, use_decode_enter={config['use_decode_enter']}")
    print(f"Others: dtype={config['dtype']}, minicpm4_yarn={config['minicpm4_yarn']}, cuda_graph={config['cuda_graph']}, memory_limit={config['memory_limit']}")
    if config['apply_sparse']:
        print(f"Sparse Attention: sink_window={config['sink_window_size']}, block_window={config['block_window_size']}, sparse_topk_k={config['sparse_topk_k']}, sparse_switch={config['sparse_switch']}, compress_lse={config['apply_compress_lse']}")
    if config['apply_eagle']:
        print(f"Eagle: eagle_num_iter={config['eagle_num_iter']}, eagle_topk_per_iter={config['eagle_topk_per_iter']}, eagle_tree_size={config['eagle_tree_size']}, apply_eagle_quant={config['apply_eagle_quant']}, window_size={config['eagle_window_size']}, frspec_vocab_size={config['frspec_vocab_size']}")
    print("=" * 50)
    print()

def main(args, config):
    if not config['test_minicpm4']:
        print(f"test_minicpm4 is False, set apply_sparse to False")
        config['apply_sparse'] = False
    
    print_config(config, config['use_stream'])
    
    # Get model paths and create model
    eagle_path, base_path, eagle_repo_id, base_repo_id = get_model_paths(args.path_prefix, config)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    llm = create_model(eagle_path, base_path, config)
    
    # Prepare input
    input_ids = make_input(tokenizer, args)
    teminators = [] if not config['use_terminators'] else [tokenizer.eos_token_id]
    
    # Initialize model
    llm.init_storage()
    
    # Apply MiniCPM4 YARN configuration if enabled
    if config['test_minicpm4'] and config['minicpm4_yarn']:
        apply_minicpm4_yarn_config(llm, config)
    
    if config['apply_eagle'] and config['frspec_vocab_size'] > 0:
        fr_path = f'{eagle_path}/freq_{config["frspec_vocab_size"]}.pt'
        if not os.path.exists(fr_path):
            cache_dir = snapshot_download(
                eagle_repo_id,
                ignore_patterns=["*.bin", "*.safetensors"],
            )
            fr_path = os.path.join(cache_dir, f'freq_{config["frspec_vocab_size"]}.pt')
        
        with open(fr_path, 'rb') as f:
            token_id_remap = torch.tensor(torch.load(f, weights_only=True), dtype=torch.int32, device="cpu")
        llm._load("token_id_remap", token_id_remap, cls="eagle")
    llm.load_from_hf()
    
    # Run generation
    if config['use_stream']:
        run_stream_generation(llm, input_ids, config, teminators, tokenizer)
    else:
        run_non_stream_generation(llm, input_ids, config, teminators, tokenizer)
    
    llm.print_perf_summary()

if __name__ == "__main__":
    args, config = parse_and_merge_config(default_config)
    main(args, config)
