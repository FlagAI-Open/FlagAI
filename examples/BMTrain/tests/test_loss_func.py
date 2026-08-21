from utils import *

import torch
import bmtrain as bmt
import torch
import random
import copy

def run(x, tgt, loss_func, bigmodel=None, scale=32768, use_float=False):
    x = x.clone().detach()
    bigmodel = copy.deepcopy(bigmodel)
    if use_float:
        x = x.float()
        if bigmodel is not None:
            bigmodel = bigmodel.float()
    x = x.requires_grad_()
    if bigmodel is None:
        loss = loss_func(x, tgt)
    else:
        t = bigmodel(x)
        loss = loss_func(t, tgt)
    (loss * scale).backward()
    return loss, x.grad

def check(x, tgt, loss_func1, loss_func2, bigmodel=None):
    loss_1, grad_1 = run(x, tgt, loss_func1, bigmodel=bigmodel)
    loss_2, grad_2 = run(x, tgt, loss_func2, bigmodel=bigmodel, use_float=True)
    assert_eq(grad_1.isnan().sum(), 0)
    assert_eq(grad_2.isnan().sum(), 0)
    print(f"{(loss_1 - loss_2).abs().item():.6f} {(grad_1 - grad_2).abs().max().item():.6f}")
    assert_lt((loss_1 - loss_2).abs().item(), 1e-5)
    assert_lt((grad_1 - grad_2).abs().max().item(), 1e-1)

def test_simple(dtype):
    loss_func1 = bmt.loss.FusedCrossEntropy()
    loss_func2 = torch.nn.CrossEntropyLoss()

    N = 32 * 512
    for i in range(1, 10):
        C = i * 10
        x = torch.randn(N, C).cuda().to(dtype)
        tgt = torch.randint(0, C, (N,)).cuda().long()
        check(x, tgt, loss_func1, loss_func2)
    for i in range(1, 10):
        C = i * 100
        x = torch.randn(N, C).cuda().to(dtype)
        tgt = torch.randint(0, C, (N,)).cuda().long()
        check(x, tgt, loss_func1, loss_func2)
    for i in range(1, 31):
        C = i * 1000
        x = torch.randn(N, C).cuda().to(dtype)
        tgt = torch.randint(0, C, (N,)).cuda().long()
        check(x, tgt, loss_func1, loss_func2)

def test_other(dtype):
    N = 32 * 512
    for i in range(1, 11):
        C = i * 10
        weight = [i+1 for i in range(C)]
        random.shuffle(weight)
        weight = torch.tensor(weight, device="cuda")
        loss_func1 = bmt.loss.FusedCrossEntropy(weight=weight.clone().to(dtype))
        loss_func2 = torch.nn.CrossEntropyLoss(weight=weight.clone().float())

        x = torch.randn(N, C).cuda().to(dtype)
        tgt = torch.randint(0, C, (N,)).cuda().long()
        mask = torch.randint(0, 2, (N,)).cuda().bool()
        tgt[mask] = -100
        check(x, tgt, loss_func1, loss_func2)

if __name__ == "__main__":
    test_other(torch.float16)
    test_simple(torch.float16)
    print("==============================================================================")
    try:
        test_other(torch.bfloat16)
        test_simple(torch.bfloat16)
    except NotImplementedError: 
        pass