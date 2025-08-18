from ... import C
from ...llm_w4a16_gptq_marlin import W4A16GPTQMarlinLLM

import numpy as np
import torch
from ..tree_drafter import *
import time
from transformers import PretrainedConfig, AutoTokenizer, AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

def pack_draft_mask(mask_2d):
    '''
    for static masks, pack them into a uint64 per row
    '''
    mask_2d_packed = torch.zeros((mask_2d.shape[0]), dtype=torch.uint16, device="cuda")
    for i in range(mask_2d.shape[0]):
        mask_1 = 0
        for j in range(i + 1):
            mask_1 |= (mask_2d[i][j].item() << j )
        mask_2d_packed[i] = mask_1
    mask_2d_packed = mask_2d_packed.view(torch.uint16).view(-1)
    return mask_2d_packed

class EagleConfig(PretrainedConfig):
    def __init__(
        self,
        num_hidden_layers=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eagle_num_layers = num_hidden_layers

class HierEagleW4A16GMSpecW4A16GM(W4A16GPTQMarlinLLM):
    def __init__(self,
                drafter_path: str,
                base_path: str,
                min_draft_length: int,
                draft_cuda_graph: bool,
                tree_path: str,
                ea_num_iter=6,
                ea_topk_per_iter=10,
                tree_size=60,
                draft_model_start=False,
                rotation=False,
                **kwargs):
        
        super().__init__(base_path, **kwargs)

        # eagle config
        self.tree_drafter_type = 'eagle'
        self.eagle_path = tree_path
        self.ea_num_iter = ea_num_iter
        self.ea_topk_per_iter = ea_topk_per_iter
        self.tree_size = tree_size

        self.tree_draft_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_position_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_gt_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_attn_mask = torch.empty((tree_size), dtype=torch.uint64, device="cuda")
        self.tree_parent = torch.empty((tree_size), dtype=torch.int32, device="cuda")


        self.eagle_config = EagleConfig.from_pretrained(self.eagle_path)
        self.rotation = rotation
        

        # draft config
        self.drafter_type = 'draft'
        self.drafter_path = drafter_path
        self.drafter_tokenizer = AutoTokenizer.from_pretrained(drafter_path)
        self.drafter_config = AutoConfig.from_pretrained(drafter_path)
        
        self.min_draft_length = min_draft_length
        self.max_draft_length = min_draft_length + ea_num_iter
        self.draft_ids = torch.empty((self.max_draft_length+1), dtype=torch.int32, device="cuda")
        self.draft_position_ids = torch.empty((self.max_draft_length+1), dtype=torch.int32, device="cuda")
        self.draft_gt_ids = torch.empty((self.max_draft_length+1), dtype=torch.int32, device="cuda")
        self.draft_attn_mask = pack_draft_mask(
            torch.tril(torch.ones(self.max_draft_length+1, self.max_draft_length+1, dtype=torch.bool)).to("cuda")
        )

        # eagle accept list
        self.draft_ea_accept_list = torch.empty((1024,), dtype=torch.int32, device="cuda")

        self.draft_logits = torch.empty((64, self.config.vocab_size), dtype=self.dtype, device="cuda")
        self.draft_cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")
        self.cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")
        self.draft_cuda_graph = draft_cuda_graph

        self.draft_model_start = draft_model_start

        self.draft_group_size = self.drafter_config.quantization_config['group_size']
        
        if self.rotation:
            C.init_hier_eagle_w4a16_gm_rot_spec_w4a16_gm_model(
                self.drafter_config.vocab_size,
                self.drafter_config.num_hidden_layers,
                self.drafter_config.hidden_size,
                self.drafter_config.intermediate_size,
                self.drafter_config.num_attention_heads,
                self.drafter_config.num_key_value_heads,
                self.drafter_config.head_dim,
                self.drafter_config.rms_norm_eps,
                self.draft_group_size,
                self.min_draft_length,
                self.draft_cuda_graph,
                self.eagle_config.eagle_num_layers,
                self.ea_num_iter,
                self.ea_topk_per_iter,
                self.tree_size,
                self.draft_model_start,
                0,
            )
        else:
            C.init_hier_eagle_w4a16_gm_spec_w4a16_gm_model(
                self.drafter_config.vocab_size,
                self.drafter_config.num_hidden_layers,
                self.drafter_config.hidden_size,
                self.drafter_config.intermediate_size,
                self.drafter_config.num_attention_heads,
                self.drafter_config.num_key_value_heads,
                self.drafter_config.head_dim,
                self.drafter_config.rms_norm_eps,
                self.draft_group_size,
                self.min_draft_length,
                self.draft_cuda_graph,
                self.eagle_config.eagle_num_layers,
                self.ea_num_iter,
                self.ea_topk_per_iter,
                self.tree_size,
                self.draft_model_start,
                0,
            )
    
    # def load_from_hf(self):
    #     self._load_from_ckpt(self.eagle_path, cls=self.tree_drafter_type)
    #     self._load_from_ckpt(self.drafter_path, cls=self.drafter_type)
    #     super().load_from_hf()

    def _load(self, name, param, dtype=None, cls=None):
        if cls == self.tree_drafter_type:
            if dtype is None:
                dtype = self.dtype
            param = param.contiguous().to(dtype)
            if (not self.rotation) and 'embed_tokens' in name:
                return
            if 'fc' in name:
                if 'weight' in name:
                    param1 = param[..., :param.shape[-1] // 2].contiguous()
                    param2 = param[..., param.shape[-1] // 2:].contiguous()
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param1.data_ptr())
                    C.load_model(f"{cls}.{name.replace('fc', 'fc2')}", param2.data_ptr())
                else: # bias
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param.data_ptr())
            else:
                C.load_model(f"{cls}.{name}", param.data_ptr())
        elif cls == self.drafter_type:
            if dtype is None:
                if 'rotary_emb' in name:
                    dtype = torch.float32
                else:
                    dtype = self.dtype

            # if 'gate_up_proj' in name:
            #     self._load(name.replace("gate_up_proj", "gate_proj"), param[:param.shape[0]//2], dtype, cls=cls)
            #     self._load(name.replace("gate_up_proj", "up_proj"), param[param.shape[0]//2:], cls=cls)
            # elif 'qkv_proj' in name:
            #     self._load(name.replace("qkv_proj", "q_proj"), param[:self.config.num_attention_heads * self.config.head_dim], cls=cls)
            #     self._load(name.replace("qkv_proj", "k_proj"), param[self.config.num_attention_heads * self.config.head_dim:(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim], cls=cls)
            #     self._load(name.replace("qkv_proj", "v_proj"), param[(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim:], cls=cls)
            # else:
            param = param.contiguous()
            if param.dtype not in [torch.int8, torch.int16, torch.int32]:
                param = param.to(dtype)
            C.load_model(f"{cls}.{name}", param.data_ptr())

            if "embed_tokens" in name and hasattr(self.config, "tie_word_embeddings") and self.config.tie_word_embeddings:
                self._load("lm_head.weight", param, cls)
        else:
            super()._load(name, param, dtype)
    
    def load_from_hf(self):
        with torch.no_grad():
            # ealge load
            self._load_from_ckpt(self.eagle_path, cls=self.tree_drafter_type)

            self._load_from_ckpt(self.drafter_path, cls=self.drafter_type)
            # rope
            if hasattr(self.drafter_config, "rope_scaling") and self.drafter_config.rope_scaling is not None:
                draft_rope_type = self.drafter_config.rope_scaling.get("rope_type", self.drafter_config.rope_scaling.get("type"))
            else:
                draft_rope_type = "default"
            # TODO only support "default", "llama3" or "longrope" with long_factor=short_factor
            draft_inv_freq, draft_attention_scaling = ROPE_INIT_FUNCTIONS[draft_rope_type](self.drafter_config, "cpu", seq_len=self.max_total_length)
            # attention_scaling = torch.tensor([attention_scaling], dtype=torch.float32, device="cpu")
            self._load(f"{self.drafter_type}.model.rotary_emb.inv_freq", draft_inv_freq, dtype=torch.float32, cls=self.drafter_type)
            # self._load("model.rotary_emb.attention_scaling", attention_scaling, dtype=torch.float32)
            
        super().load_from_hf()
    
    
    def generate(self, input_ids, generation_length=100, teminators=[]):
        assert input_ids.dtype == torch.int32
        
        prefix_length = input_ids.shape[1]
        
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        logits = self.prefill(input_ids, position_ids)
        self.draft_ids[:1].copy_(logits[0].argmax(dim=-1))

        tokens = torch.empty((generation_length), dtype=torch.int32, device="cuda")
        tokens[0].copy_(self.draft_ids[0])
        accept_lengths = []
        i = 0
        model_step = 0
        terminal = False
        torch.cuda.synchronize()
        start_time = time.time()


        while i < generation_length-1 and not terminal:
            self.cache_length[0] = prefix_length + i
            self.draft_position_ids[0] = prefix_length + i

            
            # step 1: draft model prefill and eagle input prepare
            C.draft(
                self.draft_ids.data_ptr(),
                self.draft_position_ids.data_ptr(),
                self.cache_length.data_ptr(),
                self.draft_attn_mask.data_ptr(),
                self.draft_ea_accept_list.data_ptr(),
            )
            

            # step 2: target model decode (length fixed for cuda graph)
            logits = self.decode(self.draft_ids, self.draft_position_ids, self.cache_length, mask_2d=self.draft_attn_mask)
            self.draft_gt_ids.copy_(logits.argmax(dim=-1))
            
            # step 6: verify and fix target model and eagle input
            accept_length = C.verify_and_fix(
                self.draft_ids.numel(),
                self.draft_ids.data_ptr(),
                self.draft_gt_ids.data_ptr(),
                self.draft_position_ids.data_ptr(),
                self.cache_length.data_ptr(),
                self.draft_attn_mask.data_ptr(),
                self.draft_ea_accept_list.data_ptr(),
            )

            model_step += 1
            accept_lengths.append(accept_length)
            for temin in teminators:
                if temin in self.draft_gt_ids[:accept_length]:
                    terminal = True
            append_length = min(accept_length, generation_length - 1 - i)
            tokens[1+i:1+i+append_length].copy_(self.draft_gt_ids[:append_length])
            self.draft_ids[0] = self.draft_gt_ids[accept_length - 1]
            i += accept_length
        
        
        # print(f"ea accept avg:", np.mean(self.draft_ea_accept_list[1:ea_acc_nums+1].cpu().numpy()))
        
        torch.cuda.synchronize()
        decode_time = time.time() - start_time
        ea_acc_nums = self.draft_ea_accept_list[0].item()
        tokens = tokens[:i+1].tolist()
        return tokens, accept_lengths, model_step, decode_time, self.draft_ea_accept_list[1:1+ea_acc_nums].clone()