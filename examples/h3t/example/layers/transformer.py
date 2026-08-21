from typing import Optional
import torch
import bmtrain as bmt
from layers import Layernorm, Feedforward, Attention

class TransformerEncoder(bmt.DistributedModule):
    def __init__(self,
            dim_model : int, dim_head : int, num_heads : int, dim_ff : int,
            bias : bool = True, dtype = None
        ) -> None:
        super().__init__()

        self.ln_attn = Layernorm(dim_model, dtype=dtype)
        self.attn = Attention(dim_model, dim_head, num_heads, bias=bias, dtype=dtype)

        self.ln_ff = Layernorm(dim_model, dtype=dtype)
        self.ff = Feedforward(dim_model, dim_ff, bias=bias, dtype=dtype)
    
    def forward(self,
            hidden : torch.Tensor,      # (batch, seq_len, dim_model)
            mask : torch.BoolTensor,    # (batch, seq_len, dim_model)
            position_bias : Optional[torch.Tensor] = None,   # (batch, num_head, seq_len, seq_len)
        ):
        bmt.inspect.record_tensor(hidden, "hidden")
        x = self.ln_attn(hidden)
        x = self.attn(x, x, mask, position_bias)
        hidden = hidden + x

        x = self.ln_ff(hidden)
        x = self.ff(x)
        hidden = hidden + x

        return hidden
    
