from utils import *

import torch
import torch.nn.functional as F
import bmtrain as bmt

def manual_seed(seed=33):
    torch.manual_seed(seed)
    import random as random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

class Linear_NormalInitBefore(torch.nn.Module):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        w = torch.empty(out_features, in_features, dtype=dtype, device="cuda") # use cuda to match random algorithm
        torch.nn.init.xavier_normal_(w)
        self.weight = torch.nn.Parameter(w)
        if bias:
            b = torch.empty(out_features, dtype=dtype, device="cuda") # use cuda to match random algorithm
            torch.nn.init.zeros_(b)
            self.bias = torch.nn.Parameter(b)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class Linear_NormalInitAfter(torch.nn.Module):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device="cuda")) # use cuda to match random algorithm
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=dtype, device="cuda")) # use cuda to match random algorithm
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class Linear_BMTInitializer(bmt.DistributedModule):
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

class Linear_ManualInitBefore(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        w = torch.empty(out_features, in_features, dtype=dtype, device="cuda") # use cuda to match random algorithm
        torch.nn.init.xavier_normal_(w)
        self.weight = bmt.DistributedParameter(w)
        if bias:
            b = torch.empty(out_features, dtype=dtype, device="cuda") # use cuda to match random algorithm
            torch.nn.init.zeros_(b)
            self.bias = bmt.DistributedParameter(b)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class Linear_ManualInitAfter(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=dtype, device="cuda")) # use cuda to match random algorithm
        # torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = bmt.DistributedParameter(torch.empty(out_features, dtype=dtype, device="cuda")) # use cuda to match random algorithm
            # torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Linear_Pipeline(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.l = bmt.PipelineTransformerBlockList([
            Linear_BMTInitializer(in_features, out_features, bias, dtype),
            Linear_BMTInitializer(in_features, out_features, bias, dtype),
        ])
    
    def forward(self, input):
        return self.l(input)

class Linear_BlockList(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.l = bmt.TransformerBlockList([
            Linear_BMTInitializer(in_features, out_features, bias, dtype),
            Linear_BMTInitializer(in_features, out_features, bias, dtype),
        ])
    
    def forward(self, input):
        return self.l(input)

class Linear_CheckpointList(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.l = torch.nn.ModuleList([
            bmt.CheckpointBlock(Linear_BMTInitializer(in_features, out_features, bias, dtype)),
            bmt.CheckpointBlock(Linear_BMTInitializer(in_features, out_features, bias, dtype)),
        ])
    
    def forward(self, input):
        return self.l(input)

class Linear_Checkpoint(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.l = bmt.CheckpointBlock(Linear_BMTInitializer(in_features, out_features, bias, dtype))
    
    def forward(self, input):
        return self.l(input)
    
def test_main():
    shape = [3, 5]
    # torch
    m = [None] * 10
    ret = [None] * 10
    manual_seed(33)
    m[0] = Linear_NormalInitBefore(*shape)
    ret[0] = (m[0].weight.data, m[0].bias.data)

    manual_seed(33)
    m[1] = Linear_NormalInitAfter(*shape)
    ret[1] = (m[1].weight.data, m[1].bias.data)

    # bmtrain
    manual_seed(33)
    m[2] = Linear_BMTInitializer(*shape)
    bmt.init_parameters(m[2])
    ret[2] = (m[2].weight.data, m[2].bias.data)

    manual_seed(33)
    m[3] = Linear_ManualInitBefore(*shape)
    ret[3] = (m[3].weight.data, m[3].bias.data)

    # manual_seed(33)
    # mw = Linear_ManualInitAfter(*shape) # not supported
    # print(mw.weight.data, mw.bias.data)

    manual_seed(33)
    m[4] = bmt.BMTrainModelWrapper(m[0])
    ret[4] = (m[4].weight.data, m[4].bias.data)

    manual_seed(33)
    m[5] = bmt.BMTrainModelWrapper(m[1])
    ret[5] = (m[5].weight.data, m[5].bias.data)

    manual_seed(33)
    m[6] = Linear_Pipeline(*shape)
    bmt.init_parameters(m[6])
    ret[6] = (m[6].l[0].weight.data, m[6].l[0].bias.data)

    manual_seed(33)
    m[7] = Linear_BlockList(*shape)
    bmt.init_parameters(m[7])
    ret[7] = (m[7].l[0].weight.data, m[7].l[0].bias.data)

    manual_seed(33)
    m[8] = Linear_CheckpointList(*shape)
    bmt.init_parameters(m[8])
    ret[8] = (m[8].l[0].weight.data, m[8].l[0].bias.data)

    manual_seed(33)
    m[9] = Linear_Checkpoint(*shape)
    bmt.init_parameters(m[9])
    ret[9] = (m[9].l.weight.data, m[9].l.bias.data)

    for i in range(10):
        ret[i] = ( ret[i][0].view(-1), ret[i][1].view(-1) )
        print(ret[i])
    for i in range(10):
        for j in range(10):
            assert_all_eq(ret[i][0], ret[j][0])
            assert_all_eq(ret[i][1], ret[j][1])

if __name__ == "__main__":
    bmt.init_distributed(pipe_size=1)

    test_main()