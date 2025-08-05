import torch
import bmtrain as bmt
from bmtrain.global_var import config
import numpy as np
import os

def run_normal(x, t, ckp_path, dtype):
    proj = bmt.nn.Projection(100, 64, dtype=dtype)
    bmt.init_parameters(proj)
    bmt.save(proj, ckp_path)
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, parallel=False)
    y = proj(x)
    y = y.detach().requires_grad_()
    loss = loss_func(y, t)
    loss.backward()
    return y, loss, y.grad

def run_vp(x, t, ckp_path, dtype):
    proj = bmt.nn.VPProjection(100, 64, dtype=dtype)
    bmt.load(proj, ckp_path)
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, parallel=True)
    y = proj(x)
    y = y.detach().requires_grad_()
    loss = loss_func(y, t)
    loss.backward()
    return y, loss, y.grad

def run(dtype):
    ckp_path = 'embedding.pt'
    torch.cuda.manual_seed(100)
    tp_size = config["tp_size"]
    tp_rank = config['tp_rank']
    x = torch.randn(110, 64, device='cuda', dtype=dtype)
    t = torch.cat([torch.arange(100).view(10, 10), torch.ones((10, 1))*-100], dim=-1).view(110).int().cuda()
    y1, loss1, grad1 = run_normal(x, t, ckp_path, dtype)
    y2, loss2, grad2 = run_vp(x, t, ckp_path, dtype)
    y1 = y1.chunk(tp_size, dim=-1)[tp_rank]
    grad1 = grad1.chunk(tp_size, dim=-1)[tp_rank]
    for r in range(tp_size):
        if bmt.rank() == r:
            print((y1-y2).abs().max())
            print((loss1-loss2).abs().max())
            print((grad1-grad2).abs().max())
            assert (y1-y2).abs().max() < 1e-4
            assert (loss1-loss2).abs().max() < 1e-4
            assert (grad1-grad2).abs().max() < 1e-4
        bmt.synchronize()
    if bmt.rank() == 0:
        os.remove(f"embedding.pt")

if __name__ == "__main__":
    bmt.init_distributed(tp_size=4)
    run(torch.half)
    run(torch.bfloat16)

