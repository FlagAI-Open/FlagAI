import torch
aquila_sd = torch.load('path of the aquila checkpoint')
llama_sd = {}
for i in range(0, 32):
    llama_sd[f'model.layers.{i}.self_attn.q_proj.weight'] = aquila_sd[f'transformer.layers.{i}.mixer.Wqkv.weight'][:4096]  # *
    llama_sd[f'model.layers.{i}.self_attn.k_proj.weight'] = aquila_sd[f'transformer.layers.{i}.mixer.Wqkv.weight'][4096:4096*2]  # *
    llama_sd[f'model.layers.{i}.self_attn.v_proj.weight'] = aquila_sd[f'transformer.layers.{i}.mixer.Wqkv.weight'][4096*2:]  # *
    llama_sd[f'model.layers.{i}.self_attn.o_proj.weight'] = aquila_sd[f'transformer.layers.{i}.mixer.out_proj.weight']
    llama_sd[f'model.layers.{i}.mlp.gate_proj.weight'] = aquila_sd[f'transformer.layers.{i}.mlp.fc1.weight'][11008:]  # 
    llama_sd[f'model.layers.{i}.mlp.up_proj.weight'] = aquila_sd[f'transformer.layers.{i}.mlp.fc1.weight'][:11008]  # 
    llama_sd[f'model.layers.{i}.mlp.down_proj.weight'] = aquila_sd[f'transformer.layers.{i}.mlp.fc2.weight']
    llama_sd[f'model.layers.{i}.input_layernorm.weight'] = aquila_sd[f'transformer.layers.{i}.norm1.weight']  # **
    llama_sd[f'model.layers.{i}.post_attention_layernorm.weight'] = aquila_sd[f'transformer.layers.{i}.norm2.weight']  # **
    llama_sd[f'model.layers.{i}.self_attn.rotary_emb.inv_freq'] = aquila_sd[f'transformer.layers.{i}.mixer.rotary_emb.inv_freq']
llama_sd[f'model.embed_tokens.weight'] = aquila_sd['transformer.embeddings.word_embeddings.weight']
llama_sd[f'model.norm.weight'] = aquila_sd[f'transformer.ln_f.weight']
llama_sd[f'lm_head.weight'] = aquila_sd[f'lm_head.weight']
torch.save(llama_sd, 'path to the llama checkpoint')