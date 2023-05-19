import torch

def analyze(modules, verbose=True):
    """
    modules is a list of sub-modules to search recursively. 
    
    this can be the whole model, but sometimes only some submodules want to be inspected
    """
    if verbose:
        print("\nSearching:")
        print("module | params")
    import numpy as np
    abs_min, abs_max = 1e10, 0
    norm_min, norm_max = 1e10, 0

    # when fp16
    #abs_min, abs_max = 65504, 0
    #norm_min, norm_max = 65504, 0
    for i,m in enumerate(modules):
        #for j,p in enumerate(m.parameters(recurse=True)):
        if verbose:
            print("\nParams Results:")
            print("modules   | param   | abs min   | abs max   | norm")
        for j,key in enumerate(m.keys()):
            p = m[key]
            p = p.float()
            p_abs = p.abs()
            p_abs_max = p_abs.max().item()
            p_abs_min = p_abs.min().item()
            if p_abs_min < abs_min: abs_min = p_abs_min
            if p_abs_max > abs_max: abs_max = p_abs_max
                
            p_norm = torch.linalg.norm(p.data)
            if p_norm > 0:
                if p_norm < norm_min: norm_min = p_norm
                if p_norm > norm_max: norm_max = p_norm
            if verbose:
                print(f"{i:>6} | {key} | {p_abs_min:.3e} | {p_abs_max:.3e} | {p_norm:.3e}")
    return abs_min, abs_max, norm_min, norm_max

'''
from flagai.model.llama_model import LLAMAModel
config_file = '/share/ldwang/state_dict/Aquila-7b/config.json'
model = LLAMAModel.init_from_json(config_file=config_file)

import os
cache_dir = '/share/ldwang/checkpoints/Aquila-7b-16n8g/2023050202/178000'
cache_dir = '/share/ldwang/checkpoints/Aquila-7b-16n8g/2023050202/100000'
checkpoint_path = os.path.join(cache_dir, "pytorch_model.bin")
model.load_weights(checkpoint_path)

#print(f"model {dir(model)}")

modules = model.layers
modules.append(model.tok_embeddings)
modules.append(model.norm)
'''

import torch
sd1 = torch.load('/share/ldwang/checkpoints/Aquila-7b-16n8g/2023050202/178000/pytorch_model.bin')
modules = [sd1]
abs_min, abs_max, norm_min, norm_max = analyze(modules, verbose=True)
print("\nModules Results:")
print("abs min   | abs max   | norm min  | norm max")
print(f"{abs_min:.3e} | {abs_max:.3e} | {norm_min:.3e} | {norm_max:.3e}")
sd2 = torch.load('/share/ldwang/checkpoints/Aquila-7b-16n8g/2023050202/100000/pytorch_model.bin')
modules = [sd2]
abs_min, abs_max, norm_min, norm_max = analyze(modules, verbose=True)
print("\nModules Results:")
print("abs min   | abs max   | norm min  | norm max")
print(f"{abs_min:.3e} | {abs_max:.3e} | {norm_min:.3e} | {norm_max:.3e}")
