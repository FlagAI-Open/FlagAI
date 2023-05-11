# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset
import gc
gc.collect()
torch.cuda.empty_cache()
import flash_attn
# from flagai.auto_model.auto_loader import AutoLoader

#torch.autograd.set_detect_anomaly(True)

from flash_attn.models.gpt import GPTLMHeadModel, combine_state_dicts_tp
from flash_attn.models.llama import remap_state_dict_meta_llama, llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint, state_dicts_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can input all parameters by the command line.
# For example: python train_env_trainer.py --epochs=300 --batch_size=4 --env_type=pytorch

## TODO
checkpoints = "/share/project/64node-bmt-flashatten/state_dict/"

#print('*'*20, "cache_dir", cache_dir)
#print('*'*20, "tokenizer", tokenizer)

# avoid sync loading models in case of Mem OOM
# from flagai.model.llama_model import LLAMAModel
# model = LLAMAModel.init_from_json(config_file=config_file)
model_name='30B'
checkpoint_path = '/share/projset/baaishare/baai-mrnd/llama/llama'
config = llama_config_to_gpt2_config(config_from_checkpoint(checkpoint_path, model_name))
config.vocab_size=100008
config.use_cache = False 
config.attn_pdrop = 0.0
config.resid_pdrop = 0.0
config.layer_norm_epsilon = 1e-5
config.use_flash_attn=True

config.fused_bias_fc = False
config.fused_mlp = False  # We don't have fused GatedMLP yet
config.fused_dropout_add_ln = False
config.residual_in_fp32 = False

#config.bmt = False 
#config.prenorm_std = False 
#config.prenorm = False 
#config.use_flash_attn = False
print(config)
model = GPTLMHeadModel(config, device='cpu')#,dtype= torch.float16)
print('*'*20, "model", model)
checkpoint_path = os.path.join('checkpoints/Aquila-30b-64n8g-from-scratch/20000', "pytorch_model.bin")
ckpt = torch.load(checkpoint_path)['module']
model.load_state_dict(ckpt, strict=True)
