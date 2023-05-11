from transformers import LlamaConfig, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from flash_attn.models.gpt import GPTLMHeadModel, combine_state_dicts_tp
from flash_attn.models.llama import remap_state_dict_meta_llama, llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint, state_dicts_from_checkpoint
import torch
model_name = '7B'
checkpoint_path = '/share/projset/baaishare/baai-mrnd/llama/llama'
config = llama_config_to_gpt2_config(config_from_checkpoint(checkpoint_path, model_name))
print(config)
device = torch.device('cuda:0')
model = GPTLMHeadModel(config, device=device)

batch_size = 2
max_seqlen = 256
seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device=device)
input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long,
                            device=device)
logits = model(input_ids).logits