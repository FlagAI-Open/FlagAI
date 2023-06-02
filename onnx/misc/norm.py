#!/usr/bin/env python
import torch


# 对特征进行归一化
def norm(vec):
  with torch.no_grad():
    vec /= vec.norm(dim=-1, keepdim=True)
    return vec
