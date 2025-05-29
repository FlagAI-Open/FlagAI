import torch
from collections import OrderedDict

model = torch.load("/cdgm0705/hyx/alpaca-lora/adapter_model.bin", map_location='cpu')
new_dict = OrderedDict()
# breakpoint()
for i in range(40):
    new_dict[f'model.layers.{i}.self_attn.o_proj.lora.lora_A'] = model[f'base_model.model.model.layers.{i}.self_attn.o_proj.lora_A.weight']
    new_dict[f'model.layers.{i}.self_attn.q_proj.lora.lora_A'] = model[f'base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight']
    new_dict[f'model.layers.{i}.self_attn.k_proj.lora.lora_A'] = model[f'base_model.model.model.layers.{i}.self_attn.k_proj.lora_A.weight']
    new_dict[f'model.layers.{i}.self_attn.v_proj.lora.lora_A'] = model[f'base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight']
    new_dict[f'model.layers.{i}.self_attn.o_proj.lora.lora_B'] = model[f'base_model.model.model.layers.{i}.self_attn.o_proj.lora_B.weight']
    new_dict[f'model.layers.{i}.self_attn.q_proj.lora.lora_B'] = model[f'base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight']
    new_dict[f'model.layers.{i}.self_attn.k_proj.lora.lora_B'] = model[f'base_model.model.model.layers.{i}.self_attn.k_proj.lora_B.weight']
    new_dict[f'model.layers.{i}.self_attn.v_proj.lora.lora_B'] = model[f'base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight']

    # new_dict[f'model.layers.{i}.mlp.gate_proj.lora.lora_A'] = model[f'base_model.model.model.layers.{i}.mlp.gate_proj.lora_A.weight']
    # new_dict[f'model.layers.{i}.mlp.up_proj.lora.lora_A'] = model[f'base_model.model.model.layers.{i}.mlp.up_proj.lora_A.weight']
    # new_dict[f'model.layers.{i}.mlp.down_proj.lora.lora_A'] = model[f'base_model.model.model.layers.{i}.mlp.down_proj.lora_A.weight']
    # new_dict[f'model.layers.{i}.mlp.gate_proj.lora.lora_B'] = model[f'base_model.model.model.layers.{i}.mlp.gate_proj.lora_B.weight']
    # new_dict[f'model.layers.{i}.mlp.up_proj.lora.lora_B'] = model[f'base_model.model.model.layers.{i}.mlp.up_proj.lora_B.weight']
    # new_dict[f'model.layers.{i}.mlp.down_proj.lora.lora_B'] = model[f'base_model.model.model.layers.{i}.mlp.down_proj.lora_B.weight']

torch.save(new_dict, "/cdgm0705/hyx/alpaca-lora/adapter_model.pt")