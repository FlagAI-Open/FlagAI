import torch
import bmtrain as bmt

class Block0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Block1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x,)

class Block2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x, x)

class Block10(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return [x, x, x, x, x, x, x, x, x, x]

if __name__ == "__main__":
    bmt.init_distributed()
    x = torch.tensor([1,2,3.])

    b = bmt.Block(Block0())
    y = b(x)
    assert isinstance(y, torch.Tensor)

    b = bmt.Block(Block1())
    y = b(x)
    assert isinstance(y, tuple) and len(y)==1

    b = bmt.Block(Block2())
    y = b(x)
    assert isinstance(y, tuple) and len(y)==2

    b = bmt.Block(Block10())
    y = b(x)
    assert isinstance(y, tuple) and len(y)==10
