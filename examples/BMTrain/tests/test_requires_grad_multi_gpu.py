from utils import *

import bmtrain as bmt
import torch
from bmtrain.block_layer import Block, TransformerBlockList
from bmtrain.pipe_layer import PipelineTransformerBlockList
from typing import List
import torch.nn.functional as F
from bmtrain import inspect

class Linear(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, init_weight = None, init_bias = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.out = {}
        if init_weight:
            self.weight = bmt.DistributedParameter(torch.tensor(init_weight, dtype=torch.float, device="cuda").reshape(out_features, in_features))
        else:
            self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=torch.float, device="cuda"), init_method=torch.nn.init.xavier_normal_)

        if init_bias:
            self.bias = bmt.DistributedParameter(torch.tensor(init_bias, dtype=torch.float, device="cuda").reshape(out_features,))
        else:
            self.bias = bmt.DistributedParameter(torch.empty(out_features, dtype=torch.float, device="cuda"), init_method=torch.nn.init.zeros_)
    
    def forward(self, input):
        ret = F.linear(input, self.weight, self.bias)
        return ret

def run(m, a, b):
    inp = torch.rand((1, 10, 256)).cuda()*100
    logits = m(inp)
    loss = logits.sum()
    loss.backward()

    sm = inspect.format_summary(
            inspect.inspect_model(m, '*')
        )
    return sm

def test_main():
    a = Linear(256, 256)
    b = Linear(256, 256)
    m = TransformerBlockList([Block(a), Block(b)])
    bmt.init_parameters(m)

    a.bias.requires_grad_(False)
    sm1 = run(m, a, b)
    print(sm1)
    assert_eq(sm1.split('\n')[2].split()[-2:], ["0.0000", "0.0000"])

    a.weight.requires_grad_(False)
    a.bias.requires_grad_(True)
    sm2 = run(m, a, b)
    print(sm2)
    assert_eq(sm1.split('\n')[1], sm2.split('\n')[1])
    assert_neq(sm1.split('\n')[2], sm2.split('\n')[2])

    a.weight.requires_grad_(True)
    a.bias.requires_grad_(False)
    sm3 = run(m, a, b)
    assert_neq(sm2.split('\n')[1], sm3.split('\n')[1])
    assert_eq(sm2.split('\n')[2], sm3.split('\n')[2])

def test_main_pipe():
    a = Linear(256, 256)
    b = Linear(256, 256)
    m = PipelineTransformerBlockList([Block(a), Block(b)])
    bmt.init_parameters(m)

    a.bias.requires_grad_(False)
    sm1 = run(m, a, b)
    print(sm1)
    assert_eq(sm1.split('\n')[2].split()[-2:], ["0.0000", "0.0000"])

    a.weight.requires_grad_(False)
    a.bias.requires_grad_(True)
    sm2 = run(m, a, b)
    print(sm2)
    assert_eq(sm1.split('\n')[1], sm2.split('\n')[1])
    assert_neq(sm1.split('\n')[2], sm2.split('\n')[2])

    a.weight.requires_grad_(True)
    a.bias.requires_grad_(False)
    sm3 = run(m, a, b)
    print(sm3)
    assert_neq(sm2.split('\n')[1], sm3.split('\n')[1])
    assert_eq(sm2.split('\n')[2], sm3.split('\n')[2])

if __name__ == "__main__":
    bmt.init_distributed(pipe_size=1)

    test_main()
    # test_main_pipe()
