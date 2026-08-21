from . import C

import os, json, glob
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from safetensors.torch import load_file

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
    ):
        super().__init__()

        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)
        self.dtype = dtype if dtype is not None else self.config.torch_dtype
        self.dtype_int = dtype_to_int(self.dtype)
        self.cuda_graph = cuda_graph

        self.memory_limit = int(torch.cuda.get_device_properties(0).total_memory * memory_limit)
        self.memory_pool = torch.nn.Parameter(torch.empty(self.memory_limit, dtype=torch.uint8, device="cuda"), requires_grad=False)

        self.chunk_length = chunk_length
        if not hasattr(self.config, "head_dim"):
            self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads
        C.init_base_model(
            self.memory_limit,
            self.memory_pool.data.data_ptr(),
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
        )

        self.logits = torch.empty((64, self.config.vocab_size), dtype=self.dtype, device="cuda")

    def init_storage(self):
        self.max_total_length = C.init_storage()

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
            else:
                rope_type = "default"
            # TODO only support "default", "llama3" or "longrope" with long_factor=short_factor
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](self.config, "cpu", seq_len=self.max_total_length)
            # attention_scaling = torch.tensor([attention_scaling], dtype=torch.float32, device="cpu")
            self._load("model.rotary_emb.inv_freq", inv_freq, dtype=torch.float32)
            # self._load("model.rotary_emb.attention_scaling", attention_scaling, dtype=torch.float32)

    def prefill(self, input_ids, position_ids):
        assert input_ids.dtype == torch.int32
        for i in range(0, input_ids.numel(), self.chunk_length):
            torch.cuda.nvtx.range_push(f"chunk from {i}")
            C.prefill(
                min(input_ids.numel() - i, self.chunk_length), i,
                input_ids.view(-1)[i:].data_ptr(), position_ids.view(-1)[i:].data_ptr(),
                self.logits.data_ptr()
            )
            torch.cuda.nvtx.range_pop()
        return self.logits[:1].clone()

    def decode(self, input_ids, position_ids, cache_length, mask_2d = None):
        assert input_ids.dtype == torch.int32
        assert position_ids.dtype == torch.int32
        assert cache_length.dtype == torch.int32
        if mask_2d is not None:
            assert mask_2d.dtype == torch.uint64
            assert input_ids.numel() == mask_2d.shape[0]

        torch.cuda.nvtx.range_push(f"decode")
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
        torch.cuda.nvtx.range_pop()
        return self.logits[:input_ids.numel()].clone()

    def generate(self, input_ids, generation_length=100, teminators=[]):
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        logits = self.prefill(input_ids, position_ids)
        token = logits[0].argmax(dim=-1).item()

        tokens = [token]
        if not hasattr(self, "input_ids"):
            self.input_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
            self.position_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
            self.cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")
        for i in range(generation_length-1):
            self.input_ids[0] = token
            self.position_ids[0] = prefix_length + i
            self.cache_length[0] = prefix_length + i

            logits = self.decode(self.input_ids, self.position_ids, self.cache_length)
            token = logits[0].argmax(dim=-1).item()
            tokens.append(token)
            if token in teminators:
                break
        return tokens
