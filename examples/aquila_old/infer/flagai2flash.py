# %%

import torch
import sys
import os
sys.path.append('../../../..//flagai-internal-bmt-flashatten')
from flagai.data.tokenizer import Tokenizer
from flagai.model.llama_model import LLAMAModel
model_flagai = LLAMAModel.init_from_json(config_file=config_file)

ckpt_iter = int(sys.argv[1])

# %%
# 加载Aquila7B-V1的ckpt
# ckpt_flagai = torch.load(f'{ckpt_iter}/pytorch_model.bin')

# %%
# 根据config初始化 Aquila7B 的flash attention 模型
from flash_attn.models.gpt import GPTLMHeadModel, combine_state_dicts_tp
from flash_attn.models.llama import remap_state_dict_meta_llama, llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint, state_dicts_from_checkpoint
model_name = '7B'
config = llama_config_to_gpt2_config(config_from_checkpoint("../state_dict/", "Aquila-7b"))
config.vocab_size=100008
config.use_cache = False 
config.attn_pdrop = 0.0
config.resid_pdrop = 0.0
config.layer_norm_epsilon = 1e-5

config.fused_bias_fc = False
config.fused_mlp = False  # We don't have fused GatedMLP yet
config.fused_dropout_add_ln = False
config.residual_in_fp32 = False

config.use_flash_attn = True
print(config)
model_flash = GPTLMHeadModel(config, device='cpu')#,dtype= torch.float16)

# 转化 flagai的模型权重为 flash attention的格式
# 输入是只有模型{参数名字：参数}的字典
# 如果是flashai的权重，需要注意取 ckpt['module']
def transform_flagai_to_flash(ckpt):
    tgt_ckpt = {}
    tgt_ckpt["transformer.embeddings.word_embeddings.weight"] =  ckpt.pop("tok_embeddings.weight")
    tgt_ckpt["lm_head.weight"] =  ckpt.pop("output.weight")
    tgt_ckpt['transformer.ln_f.weight'] = ckpt.pop("norm.weight")
        
    
    for l in range(32):
        # attention
        Wq = ckpt.pop(f'layers.{l}.attention.wq.weight')
        Wk = ckpt.pop(f'layers.{l}.attention.wk.weight')
        Wv = ckpt.pop(f'layers.{l}.attention.wv.weight')
        tgt_ckpt[f'transformer.layers.{l}.mixer.Wqkv.weight'] = torch.cat([Wq, Wk, Wv], dim=0)
        tgt_ckpt[f'transformer.layers.{l}.mixer.out_proj.weight'] = ckpt.pop(f'layers.{l}.attention.wo.weight')
        # feedforward
        W1 = ckpt.pop(f'layers.{l}.feed_forward.w1.weight')
        W2 = ckpt.pop(f'layers.{l}.feed_forward.w2.weight')
        W3 = ckpt.pop(f'layers.{l}.feed_forward.w3.weight')
        tgt_ckpt[f'transformer.layers.{l}.mlp.fc1.weight'] = torch.cat([W3, W1], dim=0)
        tgt_ckpt[f'transformer.layers.{l}.mlp.fc2.weight'] = W2
        # layernorm
        tgt_ckpt[f'transformer.layers.{l}.norm1.weight'] = ckpt.pop(f"layers.{l}.attention_norm.weight")
        tgt_ckpt[f'transformer.layers.{l}.norm2.weight'] = ckpt.pop(f"layers.{l}.ffn_norm.weight")
    return tgt_ckpt
ckpt_flash_ = transform_flagai_to_flash(ckpt_flagai)
model_flash.load_state_dict(ckpt_flash_, strict=True)
# %%

torch.save(ckpt_flash_, f"{ckpt_iter}-flash/pytorch_model.bin")

