from utils import *

import bmtrain as bmt
import torch

def test_main(dtype):
    x = torch.full((1,), bmt.rank() + 1, dtype=dtype, device="cuda").requires_grad_(True)
    y = bmt.distributed.all_reduce(x, "prod").view(-1)
    loss = (y * y).sum() / 2
    loss.backward()
    ref = y
    for i in range(bmt.world_size()):
        if i != bmt.rank(): ref *= i+1
    assert_eq(x.grad, ref)

def test_reducescatter():
    world_size = bmt.world_size()
    for shape in [(128,), (128,128)]:
        tensors = torch.randn(world_size, *shape, dtype=torch.half, device="cuda").requires_grad_(True)
        local_tensor = tensors[bmt.rank()]
        x = local_tensor.detach().clone().requires_grad_(True)
        y = bmt.distributed.reduce_scatter(x, "sum")
        ref = tensors.sum(0)
        partition = x.shape[0] // bmt.world_size()
        ref_p = ref[bmt.rank() * partition:(bmt.rank() + 1) * partition] 
        if bmt.rank() == 0:
            print(ref_p)
            print(y)
        assert torch.allclose(ref_p, y, atol=1e-2, rtol=1e-3)
        g = torch.randn_like(y)
        grad = torch.autograd.grad(y, x, g)[0]
        pgrad = grad[bmt.rank() * y.shape[0]: (bmt.rank() + 1) * y.shape[0]]
        ref_g = g
        if bmt.rank() == 0:
            print(ref_g)
            print(pgrad)
        assert torch.allclose(ref_g, pgrad, atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    bmt.init_distributed()
    test_reducescatter()
    test_main(torch.half)
    test_main(torch.bfloat16)
