from utils import *

import bmtrain as bmt
import torch
import random
from bmtrain import config
from bmtrain.block_layer import Block, TransformerBlockList
from bmtrain.pipe_layer import PipelineTransformerBlockList
import torch.nn.functional as F

class MultiInputReturn(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, a, b, c, d, e):
        return a*2, b+d, c*4+e*5

class Model_ZERO(torch.nn.Module):
    def __init__(self, ms) -> None:
        super().__init__()
        self.ms = TransformerBlockList([
            Block(m)
            for m in ms
        ], num_hidden=3)
    
    def forward(self, x):
        y = self.ms(*x)
        return y

class Model_PIPE(torch.nn.Module):
    def __init__(self, ms) -> None:
        super().__init__()
        self.ms = PipelineTransformerBlockList([
            Block(m)
            for m in ms
        ], num_hidden=3)
    
    def forward(self, x):
        y = self.ms(*x)
        return y

class Model_BLOCK(torch.nn.Module):
    def __init__(self, ms) -> None:
        super().__init__()
        self.ms = torch.nn.ModuleList([
            Block(m)
            for m in ms
        ])
    
    def forward(self, x):
        y = x[:3]
        other = x[3:]
        for m in self.ms:
            y = m(*y, *other)
        return y

class Model_NORMAL(torch.nn.Module):
    def __init__(self, ms) -> None:
        super().__init__()
        self.ms = torch.nn.ModuleList(ms)
    
    def forward(self, x):
        y = x[:3]
        other = x[3:]
        for m in self.ms:
            y = m(*y, *other)
        return y

def manual_seed(seed=33):
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

def run(name, cls, num_layer=4, dim=4096):
    manual_seed()

    ms = [MultiInputReturn() for i in range(num_layer)]

    inps = (
        torch.randn((dim,)).cuda(),
        torch.randn((dim,)).cuda(),
        torch.randn((dim,)).cuda(),
        torch.randn((dim,)).cuda(),
        torch.randn((dim,)).cuda(),
    )
    last_weights = (
        torch.randn((dim,)).cuda(),
        torch.randn((dim,)).cuda(),
        torch.randn((dim,)).cuda(),
    )

    for inp in inps:
        inp.requires_grad_(True)
    m = cls(ms)

    ret = ""
    logits = m(inps)
    loss = (logits[0]*last_weights[0] + logits[1]*last_weights[1] + logits[2]*last_weights[2]).sum()
    loss.backward()
    return list(logits) + [
        inp.grad
        for inp in inps
    ]

def test_main():
    ret = {}
    ret["normal"] = run("normal", Model_NORMAL)
    ret["block"] = run("block", Model_BLOCK)
    ret["zero"] = run("zero", Model_ZERO)
    # ret["pipe"] = run("pipe", Model_PIPE) # TODO pipeline not support multiple input-output yet
    for k, r in ret.items():
        bmt.print_rank(f"============={k}============")
        bmt.print_rank(r)
    for r in ret.values():
        for r2 in ret.values():
            for i in range(len(r)):
                assert_lt((r[i]-r2[i]).abs().max(), 1e-5)

if __name__ == "__main__":
    bmt.init_distributed(pipe_size=1)

    test_main()
