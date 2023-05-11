# %%

import torch
import sys
import os
sys.path.append('../../../../flagai-internal-bmt-flashatten')
from flagai.data.tokenizer import Tokenizer

# Flagai 模型初始化
config_file = "./state_dict/Aquila-7b/config.json"
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
checkpoint_path = os.path.join(f'{ckpt_iter}', "pytorch_model.bin")
ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model_flash.half()
model_flash.load_state_dict(ckpt, strict=True)

# %%
# 转化  flash attention的权重为flagai的格式
# 输入是只有模型{参数名字：参数}的字典
# 如果是flashai的权重，需要注意取 ckpt['module']
def transform_flash_to_flagai(ckpt):
    tgt_ckpt = {}
    tgt_ckpt["tok_embeddings.weight"] =  ckpt.pop("transformer.embeddings.word_embeddings.weight")
    tgt_ckpt["output.weight"] =  ckpt.pop("lm_head.weight")
    tgt_ckpt["norm.weight"] = ckpt.pop("transformer.ln_f.weight")
    
    for l in range(32):
        # attention
        Wqkv = ckpt.pop(f'transformer.layers.{l}.mixer.Wqkv.weight') 
        split_size = Wqkv.size()[0]//3
        Wq, Wk, Wv= torch.split(Wqkv,split_size)
        tgt_ckpt[f'layers.{l}.attention.wq.weight'] = Wq
        tgt_ckpt[f'layers.{l}.attention.wk.weight'] = Wk
        tgt_ckpt[f'layers.{l}.attention.wv.weight'] = Wv
        
        tgt_ckpt[f'layers.{l}.attention.wo.weight']=ckpt.pop(f'transformer.layers.{l}.mixer.out_proj.weight')
        # feedforward
        W31 = ckpt.pop(f'transformer.layers.{l}.mlp.fc1.weight')
        split_size = W31.size()[0]//2
        W3, W1= torch.split(W31,split_size)
        tgt_ckpt[f'layers.{l}.feed_forward.w1.weight'] = W1
        tgt_ckpt[f'layers.{l}.feed_forward.w3.weight'] = W3
        tgt_ckpt[f'layers.{l}.feed_forward.w2.weight'] = ckpt.pop(f'transformer.layers.{l}.mlp.fc2.weight')
        # layernorm
        tgt_ckpt[f"layers.{l}.attention_norm.weight"] = ckpt.pop(f'transformer.layers.{l}.norm1.weight') 
        tgt_ckpt[f"layers.{l}.ffn_norm.weight"] = ckpt.pop(f'transformer.layers.{l}.norm2.weight')
    return tgt_ckpt

model_flash.to('cpu')
ckpt_flagai= transform_flash_to_flagai(dict(model_flash.named_parameters()))

model_flagai.to('cuda:0').half()
model_flagai.load_state_dict(ckpt_flagai, strict=True)
torch.save(ckpt_flagai, f"{ckpt_iter}-flagai/pytorch_model.bin")
