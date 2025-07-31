from utils import *

import bmtrain as bmt
import torch

def test_main():
    x = torch.full((1,), bmt.rank() + 1, dtype=torch.half, device="cuda").requires_grad_(True)
    y = bmt.distributed.all_reduce(x, "prod").view(-1)
    loss = (y * y).sum() / 2
    loss.backward()
    ref = y
    for i in range(bmt.world_size()):
        if i != bmt.rank(): ref *= i+1
    print(x.grad)
    assert_eq(x.grad, ref)

if __name__ == "__main__":
    bmt.init_distributed()

    test_main()