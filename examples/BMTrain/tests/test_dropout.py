from utils import *

import torch
import bmtrain as bmt

class InnerModule(bmt.DistributedModule):
    def __init__(self):
        super().__init__()

        self.drop = torch.nn.Dropout(p=0.5)
    
    def forward(self, x):
        return self.drop(x)

class OutterModule(bmt.DistributedModule):
    def __init__(self) -> None:
        super().__init__()

        self.blk = bmt.TransformerBlockList([
            bmt.Block(InnerModule())
            for _ in range(5)
        ])
    
    def forward(self, x):
        return self.blk(x)

def test_main():
    model = OutterModule()

    for _ in range(5):
        model.train()
        x = torch.ones(32, device="cuda")
        y = model(x)
        print(y)
        assert_neq(x.numel()-y.nonzero().size(0), 0)

        model.eval()
        x = torch.ones(32, device="cuda")
        y = model(x)
        print(y)
        assert_eq(x.numel()-y.nonzero().size(0), 0)

if __name__ == "__main__":
    bmt.init_distributed()
    test_main()
