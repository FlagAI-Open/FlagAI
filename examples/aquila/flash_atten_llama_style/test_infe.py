import torch
import sys
from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.llama import llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint

checkpoint_path = './llama/'
config = llama_config_to_gpt2_config(config_from_checkpoint(checkpoint_path, "7B"))
config.vocab_size=100
config.use_cache = True
config.attn_pdrop = 0.0
config.resid_pdrop = 0.0
config.fused_bias_fc = False
config.layer_norm_epsilon = 1e-5

config.fused_mlp = False  # We don't have fused GatedMLP yet
config.fused_dropout_add_ln = False
config.residual_in_fp32 = False

config.bmt = False
config.prenorm = True 
config.use_flash_attn = True
#config.n_layer= 32

print(f'config {config}')

dtype = torch.float16
device = torch.device('cuda:0')
torch.cuda.set_device(device)
model = GPTLMHeadModel(config, device=device, dtype=dtype)
#ckpt = torch.load(checkpoint_path, map_location=device)
#model.load_state_dict(ckpt, strict=True)
# model.transformer.bmt_replace()
sd = model.state_dict()
torch.save(sd, 'model.pt')

model.eval()
# %%
input_ids=torch.LongTensor([[7,8]*1024]).to(device)

# %%
with torch.no_grad():
    logits_flash = model(input_ids).logits
    print(f'flash logits {logits_flash}')
    torch.save(logits_flash, 'logits_flash.pt')

def transform_flash_to_flagai(ckpt):
    tgt_ckpt = {}
    tgt_ckpt["tok_embeddings.weight"] =  ckpt.pop("transformer.embeddings.word_embeddings.weight")
    tgt_ckpt["output.weight"] =  ckpt.pop("lm_head.weight")
    tgt_ckpt["norm.weight"] = ckpt.pop("transformer.ln_f.weight")

    for l in range(config.n_layer):
        tgt_ckpt[f"layers.{l}.attention.rotary_emb.inv_freq"] = ckpt.pop(f"transformer.layers.{l}.mixer.rotary_emb.inv_freq")
    
        # attention
        Wqkv = ckpt.pop(f'transformer.layers.{l}.mixer.Wqkv.weight') 
        '''
        split_size = Wqkv.size()[0]//3
        Wq, Wk, Wv= torch.split(Wqkv,split_size)
        tgt_ckpt[f'layers.{l}.attention.wq.weight'] = Wq
        tgt_ckpt[f'layers.{l}.attention.wk.weight'] = Wk
        tgt_ckpt[f'layers.{l}.attention.wv.weight'] = Wv
        '''
        tgt_ckpt[f'layers.{l}.attention.Wqkv.weight']= Wqkv
        tgt_ckpt[f'layers.{l}.attention.wo.weight']=ckpt.pop(f'transformer.layers.{l}.mixer.out_proj.weight')

        # feedforward
        W31 = ckpt.pop(f'transformer.layers.{l}.mlp.fc1.weight')
        split_size = W31.size()[0]//2
        W3, W1= torch.split(W31,split_size)
        tgt_ckpt[f'layers.{l}.feed_forward.w1.weight'] = W1
        tgt_ckpt[f'layers.{l}.feed_forward.w3.weight'] = W3
        #tgt_ckpt[f'layers.{l}.feed_forward.fc1.weight'] = W31
        tgt_ckpt[f'layers.{l}.feed_forward.w2.weight'] = ckpt.pop(f'transformer.layers.{l}.mlp.fc2.weight')

        # layernorm
        tgt_ckpt[f"layers.{l}.attention_norm.weight"] = ckpt.pop(f'transformer.layers.{l}.norm1.weight') 
        tgt_ckpt[f"layers.{l}.ffn_norm.weight"] = ckpt.pop(f'transformer.layers.{l}.norm2.weight')
    return tgt_ckpt

config_file = "config.json"
from flagai.model.llama_model import LLAMAModel
model_flagai = LLAMAModel.init_from_json(config_file=config_file)
model_flagai.eval()
checkpoint_path = "model.pt"
ckpt = torch.load(checkpoint_path, map_location=device)
model_flagai.half()
## RotaryEmbedding should be float32
from flash_attn.layers.rotary import RotaryEmbedding
for i in range(len(model_flagai.layers)):
    model_flagai.layers[i].attention.rotary_emb = RotaryEmbedding(128.0, scale_base=None, interleaved=True)
model_flagai.load_state_dict(transform_flash_to_flagai(ckpt), strict=True)

device = torch.device('cuda:0')
model_flagai.to(device)
input_ids=torch.LongTensor([[7,8]*1024]).to(device)
with torch.no_grad():
    logits_flagai = model_flagai(input_ids,labels=input_ids)['logits']
    torch.save(logits_flagai, 'logits_flagai.pt')
    print(f'flagai logits {logits_flagai}')

