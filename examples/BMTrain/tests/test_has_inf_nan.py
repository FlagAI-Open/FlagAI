from utils import *
import torch
import bmtrain.loss._function as F
import random

def check(x, v):
    out = torch.zeros(1, dtype=torch.uint8, device="cuda")[0]
    F.has_inf_nan(x, out)
    assert_eq(out.item(), v)

def test_main(dtype):
    for i in list(range(1, 100)) + [1000]*10 + [10000]*10 + [100000]*10 + [1000000]*10:
        x = torch.rand((i,)).to(dtype).cuda()
        check(x, 0)
        p = random.randint(0, i-1)
        x[p] = x[p] / 0
        check(x, 1)
        x[p] = 2
        check(x, 0)
        p = random.randint(0, i-1)
        x[p] = 0
        x[p] = x[p] / 0
        check(x, 1)
        p = random.randint(0, i-1)
        x[p] = x[p] / 0
        p = random.randint(0, i-1)
        x[p] = x[p] / 0
        check(x, 1)
    print("That's right")

if __name__ == "__main__":
    test_main(torch.float16)
    print("==============================================================================")
    try:
        test_main(torch.bfloat16)
    except NotImplementedError: 
        pass
