import torch
import bmtrain as bmt
from layers import Linear

class Feedforward(bmt.DistributedModule):
    def __init__(self, dim_model : int, dim_ff : int, bias : bool = True, dtype = None) -> None:
        super().__init__()

        self.w_in = Linear(dim_model, dim_ff, bias = bias, dtype=dtype)
        self.w_out = Linear(dim_ff, dim_model, bias = bias, dtype=dtype)

        self.relu = torch.nn.ReLU()
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:

        return self.w_out(self.relu(self.w_in(input)))
