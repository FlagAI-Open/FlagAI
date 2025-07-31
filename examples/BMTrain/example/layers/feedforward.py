import torch
import bmtrain as bmt
from bmtrain.nn import (
    Linear,
    ColumnParallelLinear,
    RowParallelLinear)
from bmtrain.global_var import config

class Feedforward(bmt.DistributedModule):
    def __init__(self, dim_model : int, dim_ff : int, bias : bool = True, dtype = None) -> None:
        super().__init__()

        if config['tp_size'] > 1:
            self.w_in = ColumnParallelLinear(dim_model, dim_ff, bias = bias, dtype=dtype)
            self.w_out = RowParallelLinear(dim_ff, dim_model, bias = bias, dtype=dtype)
        else:
            self.w_in = Linear(dim_model, dim_ff, bias=bias, dtype=dtype)
            self.w_out = Linear(dim_ff, dim_model, bias=bias, dtype=dtype)

        self.relu = torch.nn.ReLU()
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        return self.w_out(self.relu(self.w_in(input)))
