import bmtrain as bmt
import torch
from bmtrain import config
from bmtrain.block_layer import CheckpointBlockContext,  CheckpointBlock, TransformerBlockList
from bmtrain.pipe_layer import PipelineTransformerBlockList
from typing import List
import torch.nn.functional as F
def print_rank0(s):
    if bmt.rank() == 0:
        print(s)
class Linear(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, init_weight = None, init_bias = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.out = {}
        if init_weight:
            self.weight = bmt.DistributedParameter(torch.tensor(init_weight, dtype=torch.float, device="cuda").reshape(out_features, in_features))
        else:
            self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=torch.float, device="cuda"), init_method=torch.nn.init.xavier_normal_)

        if init_bias:
            self.bias = bmt.DistributedParameter(torch.tensor(init_bias, dtype=torch.float, device="cuda").reshape(out_features,))
        else:
            self.bias = bmt.DistributedParameter(torch.empty(out_features, dtype=torch.float, device="cuda"), init_method=torch.nn.init.zeros_)
    
    def forward(self, input):
        ret = F.linear(input, self.weight, self.bias)
        return ret

def test_grad_accu():
    # normal distribute module
    m = Linear(256, 256)
    inp = torch.randn((1, 10, 256), device="cuda")
    logits = m(inp)
    loss = logits.sum()
    loss.backward()
    grad1 = m._parameters["weight"].grad.clone()
    logits = m(inp)
    loss = logits.sum()
    loss.backward()
    grad2 = m._parameters["weight"].grad
    assert torch.allclose(grad1*2, grad2)
    print_rank0("grad accumulation for distribute module passed")
    # checkpoint block
    m = CheckpointBlock(Linear(256, 256))
    inp = torch.randn((1, 10, 256), device="cuda")
    logits = m(inp)
    loss = logits.sum()
    loss.backward()
    bmt.synchronize()
    grad1 = m.weight.grad.clone()
    logits = m(inp)
    loss = logits.sum()
    loss.backward()
    bmt.synchronize()
    grad2 = m.weight.grad.clone()
    assert torch.allclose(grad1*2, grad2)
    print_rank0("grad accumulation for checkpointblock passed")
    # transformer block list
    m = TransformerBlockList([CheckpointBlock(Linear(256, 256))])
    inp = torch.randn((1, 10, 256), device="cuda")
    logits = m(inp)
    loss = logits.sum()
    loss.backward()
    bmt.synchronize()
    grad1 = m[0].weight.grad.clone()
    logits = m(inp)
    loss = logits.sum()
    loss.backward()
    bmt.synchronize()
    grad2 = m[0].weight.grad
    assert torch.allclose(grad1*2, grad2)
    print_rank0("grad accumulation for TransformerBlockList passed")


if __name__ == "__main__":
    bmt.init_distributed()
    test_grad_accu()