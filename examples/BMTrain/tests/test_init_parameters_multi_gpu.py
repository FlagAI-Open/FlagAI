from utils import *

import os
import torch
import torch.nn.functional as F
import bmtrain as bmt

def manual_seed(seed=33):
    torch.manual_seed(seed)
    import random as random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

class Linear_Normal(torch.nn.Module):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device="cuda")) # use cuda to match random algorithm
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=dtype, device="cuda")) # use cuda to match random algorithm
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class Linear_BMTInitializer(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=dtype), init_method=torch.nn.init.xavier_normal_)
        if bias:
            self.bias = bmt.DistributedParameter(torch.empty(out_features, dtype=dtype), init_method=torch.nn.init.zeros_)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class Linear_NormalList(torch.nn.Module):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.l = torch.nn.ModuleList([
            Linear_Normal(in_features, out_features, bias, dtype),
            Linear_Normal(in_features, out_features, bias, dtype),
        ])
    
    def forward(self, input):
        return self.l(input)

class Linear_Pipeline(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.l = bmt.PipelineTransformerBlockList([
            Linear_BMTInitializer(in_features, out_features, bias, dtype),
            Linear_BMTInitializer(in_features, out_features, bias, dtype),
        ])
    
    def forward(self, input):
        return self.l(input)

class Linear_BlockList(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.l = bmt.TransformerBlockList([
            Linear_BMTInitializer(in_features, out_features, bias, dtype),
            Linear_BMTInitializer(in_features, out_features, bias, dtype),
        ])
    
    def forward(self, input):
        return self.l(input)

class Linear_CheckpointList(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.l = torch.nn.ModuleList([
            bmt.CheckpointBlock(Linear_BMTInitializer(in_features, out_features, bias, dtype)),
            bmt.CheckpointBlock(Linear_BMTInitializer(in_features, out_features, bias, dtype)),
        ])
    
    def forward(self, input):
        return self.l(input)

def check(ckpt_path, ckpt_path_ref):
    if bmt.rank() == 0:
        ckpt1 = torch.load(ckpt_path)
        ckpt2 = torch.load(ckpt_path_ref)
        for (k1, v1), (k2, v2) in zip(ckpt1.items(), ckpt2.items()):
            assert_eq(k1, k2)
            print(v1, v2)
            assert_all_eq(v1.cuda(), v2.cuda())
    
def test_main():
    ckpt_path_ref = "test_ckpt_ref.pt"
    ckpt_path = "test_ckpt.pt"
    shape = [3, 5]
    # torch
    m = [None] * 4
    ret = [None] * 4

    manual_seed(33)
    m[0] = Linear_NormalList(*shape)
    if bmt.rank() == 0:
        torch.save(m[0].state_dict(), ckpt_path_ref)

    manual_seed(33)
    m[1] = Linear_Pipeline(*shape)
    bmt.init_parameters(m[1])
    bmt.save(m[1], ckpt_path)
    check(ckpt_path, ckpt_path_ref)

    # bmtrain
    manual_seed(33)
    m[2] = Linear_BlockList(*shape)
    bmt.init_parameters(m[2])
    bmt.save(m[2], ckpt_path)
    check(ckpt_path, ckpt_path_ref)

    manual_seed(33)
    m[3] = Linear_CheckpointList(*shape)
    bmt.init_parameters(m[3])
    bmt.save(m[3], ckpt_path)
    check(ckpt_path, ckpt_path_ref)

    if bmt.rank() == 0:
        os.remove(ckpt_path)
        os.remove(ckpt_path_ref)

if __name__ == "__main__":
    bmt.init_distributed(pipe_size=2)

    test_main()