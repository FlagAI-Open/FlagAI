import torch
import math

def glorot_normal(tensor: torch.Tensor):
    return torch.nn.init.xavier_normal_(tensor, gain=math.sqrt(2))

def glorot_uniform(tensor: torch.Tensor):
    return torch.nn.init.xavier_uniform_(tensor, gain=math.sqrt(2))
