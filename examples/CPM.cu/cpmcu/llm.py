from . import C

import os, json, glob
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from safetensors.torch import load_file
import time, math
import torch.nn.functional as F

dtype_map = {
    torch.float16: 0,
    torch.bfloat16: 1,
}

def dtype_to_int(dtype):
    ret = dtype_map.get(dtype, -1)
    if ret == -1:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return ret

class LLM(torch.nn.Module):
    def __init__(self,
                 path: str, # hf model path
                 memory_limit: float = 0.8,
                 chunk_length: int = 1024,
                 dtype: torch.dtype = None,
                 cuda_graph: bool = False,
                 apply_sparse: bool = False,
                 sink_window_size: int = 1,
                 block_window_size: int = 32,
                 sparse_topk_k: int = 32,
                 sparse_switch: int = 8192,
                 apply_compress_lse: bool = False,
                 use_enter: bool = False,
                 use_decode_enter: bool = False,
                 temperature: float = 0.0,
                 random_seed: int = None,
    ):
        super().__init__()

        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        self.dtype = dtype if dtype is not None else self.config.torch_dtype
        self.dtype_int = dtype_to_int(self.dtype)
        self.cuda_graph = cuda_graph
        self.use_enter = use_enter
        self.use_decode_enter = use_decode_enter
        self.temperature = temperature

        self.chunk_length = chunk_length
        # Flag for showing prefill progress (used in stream mode)
        self._show_prefill_progress = False
        
        # Initialize random generator if random_seed is provided
        if random_seed is not None:
            self.generator = torch.Generator(device="cuda")
            self.generator.manual_seed(random_seed)
        else:
            self.generator = None
        
        if not hasattr(self.config, "head_dim"):
            self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads
        scale_embed = self.config.scale_emb if hasattr(self.config, "scale_emb") else 1.0
        scale_lmhead = (self.config.dim_model_base / self.config.hidden_size) if hasattr(self.config, "dim_model_base") else 1.0
        scale_residual = self.config.scale_depth / math.sqrt(self.config.num_hidden_layers) if hasattr(self.config, "scale_depth") else 1.0

        if apply_sparse:
            C.init_minicpm4_model(
                memory_limit,
                self.config.vocab_size,
                self.config.num_hidden_layers,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                self.config.rms_norm_eps,
                self.dtype_int,
                self.chunk_length,
                scale_embed,
                scale_lmhead,
                scale_residual,
                sink_window_size,
                block_window_size,
                sparse_topk_k,
                sparse_switch,
                apply_compress_lse,
            )
        else:
            C.init_base_model(
                memory_limit,
                self.config.vocab_size,
                self.config.num_hidden_layers,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                self.config.rms_norm_eps,
                self.dtype_int,
                self.chunk_length,
                scale_embed,
                scale_lmhead,
                scale_residual,
            )

        self.logits = torch.empty((64, self.config.vocab_size), dtype=self.dtype, device="cuda")

    def init_storage(self):
        self.max_total_length = C.init_storage()
        print("max supported length under current memory limit: ", self.max_total_length)

    def _load(self, name, param, dtype=None, cls=None):
        if dtype is None:
            if 'rotary_emb' in name:
                dtype = torch.float32
            else:
                dtype = self.dtype

        if 'gate_up_proj' in name:
            self._load(name.replace("gate_up_proj", "gate_proj"), param[:param.shape[0]//2], dtype)
            self._load(name.replace("gate_up_proj", "up_proj"), param[param.shape[0]//2:])
        elif 'qkv_proj' in name:
            self._load(name.replace("qkv_proj", "q_proj"), param[:self.config.num_attention_heads * self.config.head_dim])
            self._load(name.replace("qkv_proj", "k_proj"), param[self.config.num_attention_heads * self.config.head_dim:(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim])
            self._load(name.replace("qkv_proj", "v_proj"), param[(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim:])
        else:
            param = param.contiguous().to(dtype)
            C.load_model(name, param.data_ptr())

        if "embed_tokens" in name and hasattr(self.config, "tie_word_embeddings") and self.config.tie_word_embeddings:
            self._load("lm_head.weight", param)

    def _load_from_ckpt(self, path, cls=None):
        supported_suffix_1 = ["bin.index.json", "safetensors.index.json"]
        supported_suffix_2 = ["bin", "safetensors", "pt"]
        file = None
        for suffix in supported_suffix_1:
            files = glob.glob(os.path.join(path, f"*.{suffix}"))
            if len(files) > 1:
                raise ValueError(f"Multiple files with suffix {suffix} found in {path}")
            elif len(files) == 1:
                file = files[0]
                break
        else:
            for suffix in supported_suffix_2:
                files = glob.glob(os.path.join(path, f"*.{suffix}"))
                if len(files) > 1:
                    raise ValueError(f"Multiple files with suffix {suffix} found in {path}")
                elif len(files) == 1:
                    file = files[0]
                    break
            else:
                raise ValueError(f"No supported checkpoint file found in {path}, supported suffixes: {supported_suffix_1 + supported_suffix_2}")

        if file.endswith(".index.json"):
            with open(file, "r") as f:
                file_list = set(json.load(f)["weight_map"].values())
            file_list = [os.path.join(path, file) for file in file_list]
        else:
            file_list = [file]

        for file in file_list:
            print(f"load from {file}")
            if file.endswith(".bin") or file.endswith(".pt"):
                ckpt = torch.load(file, map_location="cpu")
            elif file.endswith(".safetensors"):
                ckpt = load_file(file)
            for name, param in ckpt.items():
                self._load(name, param, cls=cls)

    def load_from_hf(self):
        with torch.no_grad():
            self._load_from_ckpt(self.path)

            # rope
            if hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
                rope_type = self.config.rope_scaling.get("rope_type", self.config.rope_scaling.get("type"))
                if rope_type == "longrope" and not hasattr(self.config.rope_scaling, "factor"):
                    self.config.rope_scaling["factor"] = 1.0
            else:
                rope_type = "default"
            # TODO only support "default", "llama3" or "longrope" with long_factor=short_factor
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](self.config, "cpu", seq_len=self.max_total_length)
            # attention_scaling = torch.tensor([attention_scaling], dtype=torch.float32, device="cpu")
            self._load("model.rotary_emb.inv_freq", inv_freq, dtype=torch.float32)
            # self._load("model.rotary_emb.attention_scaling", attention_scaling, dtype=torch.float32)

    def prefill(self, input_ids, position_ids):
        assert input_ids.dtype == torch.int32
        # Check if input length exceeds maximum supported length
        if input_ids.numel() > self.max_total_length:
            raise ValueError(f"Input token count ({input_ids.numel()}) exceeds maximum supported length ({self.max_total_length}) under current memory limit")
        
        total_length = input_ids.numel()
        num_chunks = (total_length + self.chunk_length - 1) // self.chunk_length
        
        prefill_start_time = None
        actual_prefill_start = None
        
        # User interaction logic only when use_enter is True
        if self._show_prefill_progress and self.use_enter:
            # Clear screen and move cursor to top, then show prompt and wait for user input
            print("\033[2J\033[H", end="", flush=True)  # Clear screen and move to top
            print("Please Press Enter to Start Prefilling...", end="", flush=True)
            input()  # Wait for Enter key
            
            # Replace the prompt with [Prefilling] - clear entire line first
            print("\r" + " " * 50 + "\r[Prefilling]", flush=True)
            # Start timing after user presses Enter
            prefill_start_time = time.time()
            actual_prefill_start = prefill_start_time
        
        # Initialize progress display for stream mode (always when _show_prefill_progress is True)
        if self._show_prefill_progress:
            if prefill_start_time is None:  # Only set start time if not already set above
                prefill_start_time = time.time()
            if not self.use_enter:
                print("Prefilling: 0.0% (0/{} tokens) @ 0.0 tokens/s".format(total_length), end="", flush=True)
            else:
                print("Prefilling: 0.0% (0/{} tokens) @ 0.0 tokens/s".format(total_length), end="", flush=True)
        
        # Record actual computation start time if not set yet
        if actual_prefill_start is None:
            actual_prefill_start = time.time()
        
        for chunk_idx, i in enumerate(range(0, input_ids.numel(), self.chunk_length)):
            # torch.cuda.nvtx.range_push(f"chunk from {i}")
            C.prefill(
                min(input_ids.numel() - i, self.chunk_length), i,
                input_ids.view(-1)[i:].data_ptr(), position_ids.view(-1)[i:].data_ptr(),
                self.logits.data_ptr()
            )
            # torch.cuda.nvtx.range_pop()
            
            # Show progress for stream mode - always when _show_prefill_progress is True
            if self._show_prefill_progress and prefill_start_time is not None:
                current_tokens = min(i + self.chunk_length, total_length)
                elapsed_time = time.time() - prefill_start_time
                progress = (current_tokens * 100.0) / total_length
                tokens_per_sec = current_tokens / elapsed_time if elapsed_time > 0 else 0.0
                print(f"\rPrefilling: {progress:.1f}% ({current_tokens}/{total_length} tokens) @ {tokens_per_sec:.1f} tokens/s", end="", flush=True)
        
        # Calculate actual prefill time
        actual_prefill_time = time.time() - actual_prefill_start
        
        # Final completion status for stream mode
        if self._show_prefill_progress:
            if prefill_start_time is not None:
                final_elapsed_time = time.time() - prefill_start_time
                final_tokens_per_sec = total_length / final_elapsed_time if final_elapsed_time > 0 else 0.0
                print(f"\rPrefilling: 100.0% ({total_length}/{total_length} tokens) @ {final_tokens_per_sec:.1f} tokens/s - Complete!")
            if self.use_enter:
                print("\n[Decoding]")  # Show decoding status and move to next line only with use_enter
            else:
                print()  # Just a newline for normal mode
        
        # Store the actual prefill time for use in generate method
        self._last_prefill_time = actual_prefill_time
        
        return self.logits[:1].clone()

    def decode(self, input_ids, position_ids, cache_length, mask_2d = None):
        assert input_ids.dtype == torch.int32
        assert position_ids.dtype == torch.int32
        assert cache_length.dtype == torch.int32
        if mask_2d is not None:
            # assert mask_2d.dtype == torch.uint64
            assert input_ids.numel() == mask_2d.shape[0]

        # torch.cuda.nvtx.range_push(f"decode")
        cache_length += input_ids.numel() # temparary add for convinience in flash_attn
        padded_length = (cache_length[0].item() + 128 - 1) // 128 * 128
        C.decode(
            input_ids.numel(), padded_length,
            input_ids.data_ptr(), position_ids.data_ptr(), cache_length.data_ptr(),
            mask_2d.data_ptr() if mask_2d is not None else 0,
            self.logits.data_ptr(),
            self.cuda_graph
        )
        cache_length -= input_ids.numel()
        # torch.cuda.nvtx.range_pop()
        return self.logits[:input_ids.numel()].clone()

    def generate(self, input_ids, generation_length=100, teminators=[], use_stream=False):
        """
        Generate text with optional streaming output.
        Returns (tokens, decode_time, prefill_time) if use_stream=False, or generator yielding {'token', 'text', 'is_finished', 'prefill_time', 'decode_time'} if use_stream=True.
        """
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        
        # Set progress flag before prefill for stream mode
        if use_stream:
            self._show_prefill_progress = True
        
        # Measure prefill time
        if self.use_enter and use_stream:
            # In use_enter mode, timing will be handled inside prefill method
            logits = self.prefill(input_ids, position_ids)
            prefill_time = getattr(self, '_last_prefill_time', 0.0)  # Get actual prefill time
        else:
            torch.cuda.synchronize()
            prefill_start = time.time()
            logits = self.prefill(input_ids, position_ids)
            torch.cuda.synchronize()
            prefill_time = time.time() - prefill_start
        
        if self.temperature > 0.0:
            token = torch.multinomial(F.softmax(logits[0]/self.temperature, dim=-1), num_samples=1, generator=self.generator)[0].item()
        else:
            token = logits[0].argmax(dim=-1).item()

        # Wait for user input before decode phase if use_decode_enter is enabled
        if self.use_decode_enter:
            if use_stream and self.use_enter:
                # In stream mode with use_enter, we already showed [Decoding], just wait for input
                print("Please Press Enter to Start Decoding...", end="", flush=True)
                input()  # Wait for Enter key
                print("\r" + " " * 50 + "\r", end="", flush=True)  # Clear the prompt without showing [Decoding] again
            else:
                # In other modes, show prompt and wait
                print("Please Press Enter to Start Decoding...", end="", flush=True)
                input()  # Wait for Enter key
                print("\r" + " " * 50 + "\r[Decoding]", flush=True)  # Show [Decoding] only when use_enter is not enabled

        if not hasattr(self, "input_ids"):
            self.input_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
            self.position_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
            self.cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")

        if use_stream:
            # Stream generation (optimized)
            def _stream_generator():
                nonlocal token
                # Keep minimal context for correct spacing
                prev_token = token
                
                # yield first token
                text = self.tokenizer.decode([token], skip_special_tokens=False)
                
                yield {
                    'token': token,
                    'text': text,
                    'is_finished': token in teminators,
                    'prefill_time': prefill_time,
                    'decode_time': 0.0  # First token comes from prefill
                }
                
                if token in teminators:
                    return

                decode_start_time = time.time()

                for i in range(generation_length-1):
                    self.input_ids[0] = token
                    self.position_ids[0] = prefix_length + i
                    self.cache_length[0] = prefix_length + i

                    logits = self.decode(self.input_ids, self.position_ids, self.cache_length)
                    if self.temperature > 0.0:
                        token = torch.multinomial(F.softmax(logits[0]/self.temperature, dim=-1), num_samples=1, generator=self.generator)[0].item()
                    else:
                        token = logits[0].argmax(dim=-1).item()
                    
                    # For correct spacing, decode with previous token context
                    if prev_token is not None:
                        context_tokens = [prev_token, token]
                        context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=False)
                        prev_text = self.tokenizer.decode([prev_token], skip_special_tokens=False)
                        text = context_text[len(prev_text):]
                    else:
                        text = self.tokenizer.decode([token], skip_special_tokens=False)
                    
                    is_finished = token in teminators or i == generation_length - 2
                    
                    # Calculate time only when needed to reduce overhead
                    decode_time = time.time() - decode_start_time
                        
                    yield {
                        'token': token,
                        'text': text,
                        'is_finished': is_finished,
                        'prefill_time': 0.0,  # Only report prefill_time for first token
                        'decode_time': decode_time
                    }
                    
                    if token in teminators:
                        break
                    
                    # Update prev_token
                    prev_token = token
            
            return _stream_generator()
        else:
            # Original batch generation
            tokens = [token]
            torch.cuda.synchronize()
            decode_start = time.time()
            for i in range(generation_length-1):
                self.input_ids[0] = token
                self.position_ids[0] = prefix_length + i
                self.cache_length[0] = prefix_length + i

                logits = self.decode(self.input_ids, self.position_ids, self.cache_length)
                if self.temperature > 0.0:
                    token = torch.multinomial(F.softmax(logits[0]/self.temperature, dim=-1), num_samples=1, generator=self.generator)[0].item()
                else:
                    token = logits[0].argmax(dim=-1).item()
                tokens.append(token)
                if token in teminators:
                    break
            torch.cuda.synchronize()
            decode_time = time.time() - decode_start
            return tokens, decode_time, prefill_time

    def print_perf_summary(self):
        C.print_perf_summary()