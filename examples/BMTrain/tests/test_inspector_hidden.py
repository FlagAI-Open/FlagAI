from utils import *

import bmtrain as bmt
import random
import torch
from bmtrain import config
from bmtrain.block_layer import Block, TransformerBlockList
from bmtrain.pipe_layer import PipelineTransformerBlockList
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

class L2(bmt.DistributedModule):
    def __init__(self, dim : int):
        super().__init__()
        self.m1 = Linear(dim, dim)
        self.m2 = Linear(dim, dim)

    def forward(self, x):
        x = self.m1(x)
        x = self.m2(x)
        return x

class L2_record(bmt.DistributedModule):
    def __init__(self, dim : int):
        super().__init__()
        self.m1 = Linear(dim, dim)
        self.m2 = Linear(dim, dim)

    def forward(self, x):
        x = self.m1(x)
        inspect.record_tensor(x, "hidden")
        x = self.m2(x)
        return x

class Model_ZERO(torch.nn.Module):
    def __init__(self, pre, ms, post) -> None:
        super().__init__()
        self.pre = pre
        self.ms = TransformerBlockList([
            Block(m)
            for m in ms
        ])
        self.post = post
    
    def forward(self, x, return_hidden_states=False):
        x = self.pre(x)
        if return_hidden_states:
            x, o = self.ms(x, return_hidden_states=return_hidden_states)
            return self.post(x), o
        else:
            x = self.ms(x, return_hidden_states=return_hidden_states)
            return self.post(x)

class Model_PIPE(torch.nn.Module):
    def __init__(self, pre, ms, post) -> None:
        super().__init__()
        self.pre = pre
        self.ms = PipelineTransformerBlockList([
            Block(m)
            for m in ms
        ])
        self.post = post
    
    def forward(self, x, return_hidden_states=False):
        x = self.pre(x)
        if return_hidden_states:
            x, o = self.ms(x, return_hidden_states=return_hidden_states)
            return self.post(x), o
        else:
            x = self.ms(x, return_hidden_states=return_hidden_states)
            return self.post(x)

class Model_BLOCK(torch.nn.Module):
    def __init__(self, pre, ms, post) -> None:
        super().__init__()
        self.pre = pre
        self.ms = torch.nn.ModuleList([
            Block(m)
            for m in ms
        ])
        self.post = post
    
    def forward(self, x, return_hidden_states=False):
        x = self.pre(x)
        o = []
        y = x
        for m in self.ms:
            o.append(y)
            y = m(y)
        if return_hidden_states:
            return self.post(y), o
        else:
            return self.post(y)

class Model_NORMAL(torch.nn.Module):
    def __init__(self, pre, ms, post) -> None:
        super().__init__()
        self.pre = pre
        self.ms = torch.nn.ModuleList(ms)
        self.post = post
    
    def forward(self, x, return_hidden_states=False):
        x = self.pre(x)
        o = []
        y = x
        for m in self.ms:
            o.append(y)
            y = m(y)
        if return_hidden_states:
            return self.post(y), o
        else:
            return self.post(y)

def manual_seed(seed=33):
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

def sub_run(name, cls, num_layer, dim, batch, seq_len):
    manual_seed()

    pre = Linear(dim, dim)
    post = Linear(dim, dim)
    ms = [L2_record(dim) if i%2==0 else L2(dim) for i in range(num_layer)]

    inp = torch.randn((batch, seq_len, dim)).cuda()
    last_weight = torch.randn((batch, seq_len, dim)).cuda()
    middle_weight = [
        torch.randn((batch, seq_len, dim)).cuda()
        for i in range(len(ms)//2)
    ]

    bmt.init_parameters(pre)
    bmt.init_parameters(post)
    for m in ms:
        bmt.init_parameters(m)
    m = cls(pre, [m for m in ms], post)
    ret = ""
    with inspect.inspect_tensor() as inspector:
        logits = m(inp)

    ret += inspect.format_summary(
        inspector.get_summary()
    ) + "\n"

    loss = 0
    for i in range(len(ms)//2):
        loss = loss + (inspector.summary[i]['tensor'] * middle_weight[i]).sum()

    with inspect.inspect_tensor():
        loss.backward()

    ret += inspect.format_summary(
        inspect.inspect_model(m, '*')
    ) + "\n"
    ret += inspect.format_summary(
        inspector.get_summary()
    ) + "\n"

    with inspect.inspect_tensor() as inspector:
        logits = m(inp)

    ret += inspect.format_summary(
        inspector.get_summary()
    ) + "\n"

    loss = (logits * last_weight).sum()

    with inspect.inspect_tensor():
        loss.backward()

    ret += inspect.format_summary(
        inspect.inspect_model(m, '*')
    ) + "\n"
    ret += inspect.format_summary(
        inspector.get_summary()
    ) + "\n"

    return ret + "\n" # replace for matching None grad with zero_grad

def run(name, cls, num_layer=4, dim=4096, batch=32, seq_len=256):
    ret = ""
    ret += sub_run(name, cls, num_layer=num_layer, dim=dim, batch=batch, seq_len=seq_len)
    bmt.synchronize()
    return ret

def test_main():
    ret = {}
    ret["normal"] = run("normal", Model_NORMAL)
    ret["block"] = run("block", Model_BLOCK)
    ret["zero"] = run("zero", Model_ZERO)
    ret["pipe"] = run("pipe", Model_PIPE)
    for k, r in ret.items():
        bmt.print_rank(f"============={k}============")
        bmt.print_rank(r)
    for r in ret.values():
        for r2 in ret.values():
            lines, lines2 = r.split('\n'), r2.split('\n')
            assert len(lines) == len(lines2)
            for line, line2 in zip(lines, lines2):
                words, words2 = line.split(), line2.split()
                assert len(words) == len(words2)
                for w, w2 in zip(words, words2):
                    try:
                        is_float = isinstance(eval(w), float)
                    except:
                        is_float = False
                    if is_float:
                        assert_lt(abs(float(w)-float(w2)), 0.00011)
                    else:
                        assert_eq(w, w2)

if __name__ == "__main__":
    bmt.init_distributed(pipe_size=2)

    test_main()
