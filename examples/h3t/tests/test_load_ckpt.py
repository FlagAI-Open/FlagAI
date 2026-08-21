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
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=dtype))
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
    m = Linear_Normal(7, 5).cuda()
    m2 = Linear_BMT(7, 5)
    torch.save(m.state_dict(), ckpt_path)
    bmt.load(m2, ckpt_path)

    print(m.weight.data)
    print(m2.weight.data)
    assert_all_eq(m.weight.data, m2.weight.data)
    assert_all_eq(m.bias.data, m2.bias.data)

    os.remove(ckpt_path)

    bmt.save(m2, ckpt_path)
    m.load_state_dict(torch.load(ckpt_path))

    print(m.weight.data)
    print(m2.weight.data)
    assert_all_eq(m.weight.data, m2.weight.data)
    assert_all_eq(m.bias.data, m2.bias.data)

    os.remove(ckpt_path)

if __name__ == "__main__":
    bmt.init_distributed()

    test_main()