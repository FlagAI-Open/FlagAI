from utils import *
import torch
import torch.nn.functional as F
import bmtrain as bmt
import os

class Linear_Normal(torch.nn.Module):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device="cuda"))
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=dtype, device="cuda"))
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class Linear_BMT(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=dtype), init_method=torch.nn.init.xavier_normal_)
        if bias:
            self.bias = bmt.DistributedParameter(torch.empty(out_features, dtype=dtype), init_method=torch.nn.init.zeros_)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


def test_main():
    ckpt_path = "test_ckpt.pt"
    # Transformer BlockList
    m = Linear_Normal(256, 256).cuda()
    m2 = bmt.TransformerBlockList([bmt.Block(Linear_BMT(256, 256))])
    if bmt.rank() == 0:
        torch.save(m.state_dict(), ckpt_path)
    dic2 = m.state_dict()
    dic2["0.weight"] = dic2.pop("weight")
    dic2["0.bias"] = dic2.pop("bias")
    m2.load_state_dict(dic2)
    for key in m.state_dict():
        bmt_key = f"0.{key}"
        assert bmt_key in m2.state_dict(), "wrong key in bmtrain model"
        assert (m2.state_dict()[bmt_key].cuda() == m.state_dict()[key]).all() , "wrong param in bmtrain model"
    if bmt.rank() == 0:
        os.remove(ckpt_path)
    print("Transformer Blocklist load_state_dict and state_dict test passed")

    # Block 
    m3 = bmt.Block(Linear_BMT(256, 256))
    m3.load_state_dict(m.state_dict())
    for key in m.state_dict():
        assert key in m3.state_dict(), "wrong key in bmtrain model"
        assert (m.state_dict()[key] == m3.state_dict()[key].cuda()).all(), "wrong param in bmtrain model"
    print("Block load_state_dict and state_dict test passed")

    # normal Distributed module
    m4 = Linear_BMT(256, 256)
    m4.load_state_dict(m.state_dict())
    for key in m.state_dict():
        assert key in m4.state_dict(), "wrong key in bmtrain model"
        assert (m.state_dict()[key] == m4.state_dict()[key].cuda()).all(), "wrong param in bmtrain model"
    print("bmt.distributedmodule load_state_dict and state_dict test passed")

if __name__ == "__main__":
    bmt.init_distributed()

    test_main()
