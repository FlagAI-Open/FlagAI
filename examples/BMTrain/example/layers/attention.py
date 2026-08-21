from typing import Optional
import torch
import bmtrain as bmt
from bmtrain.nn import (
    Linear,
    ColumnParallelLinear,
    RowParallelLinear,
)
import math
from bmtrain.global_var import config
from bmtrain.distributed import all_gather 

class Attention(bmt.DistributedModule):
    def __init__(self, 
            dim_model : int, dim_head : int,
            num_heads : int, bias : bool = True,
            dtype = None
        ) -> None:
        super().__init__()

        if config['tp_size'] > 1:
            self.project_q = ColumnParallelLinear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype, gather_input=False)
            self.project_k = ColumnParallelLinear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype, gather_input=False)
            self.project_v = ColumnParallelLinear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype, gather_input=False)
            self.project_out = RowParallelLinear(dim_head * num_heads, dim_model, bias=bias, dtype=dtype)
        else:
            self.project_q = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
            self.project_k = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
            self.project_v = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
            self.project_out = Linear(dim_head * num_heads, dim_model, bias=bias, dtype=dtype)


        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_model = dim_model
    
    def forward(self, 
            hidden_q : torch.Tensor,        # (batch_size, seq_q, dim_model)
            hidden_kv : torch.Tensor,       # (batch_size, seq_kv, dim_model)
            mask : torch.BoolTensor,        # (batch_size, seq_q, seq_kv)
            position_bias : Optional[torch.Tensor] = None,   # (batch, num_heads, seq_q, seq_kv)
        ) -> torch.Tensor:
        batch_size = hidden_q.size()[0]

        assert hidden_q.data_ptr() == hidden_kv.data_ptr()

        if config['tp_size'] > 1:
            hidden_q = bmt.nn.OpParallelLinear.apply(
                hidden_q,
                torch.cat([self.project_q.weight, self.project_k.weight, self.project_v.weight], dim=0),
                torch.cat([self.project_q.bias, self.project_k.bias, self.project_v.bias], dim=0),
                True, False,
                False, None
            )
            hidden_q = hidden_q.view(batch_size, -1, hidden_q.shape[-1])
            h_q, h_k, h_v = hidden_q.chunk(3, dim=-1)
        else:
            h_q : torch.Tensor = self.project_q(hidden_q)
            h_k : torch.Tensor = self.project_k(hidden_kv)
            h_v : torch.Tensor = self.project_v(hidden_kv)

        seq_q = h_q.size()[1]
        seq_kv = h_k.size(1)

        h_q = h_q.view(batch_size, seq_q, -1, self.dim_head)
        h_k = h_k.view(batch_size, seq_kv, -1, self.dim_head)
        h_v = h_v.view(batch_size, seq_kv, -1, self.dim_head)

        h_q = h_q.permute(0, 2, 1, 3).contiguous()
        h_k = h_k.permute(0, 2, 1, 3).contiguous()
        h_v = h_v.permute(0, 2, 1, 3).contiguous()

        h_q = h_q.view(-1, seq_q, self.dim_head)
        h_k = h_k.view(-1, seq_kv, self.dim_head)
        h_v = h_v.view(-1, seq_kv, self.dim_head)

        score = torch.bmm(
            h_q, h_k.transpose(1, 2)
        )
        score = score / math.sqrt(self.dim_head)

        score = score.view(batch_size, -1, seq_q, seq_kv)

        if position_bias is not None:
            score = score + position_bias.view(batch_size, -1, seq_q, seq_kv)
         
        score = torch.where(
            mask.view(batch_size, 1, seq_q, seq_kv),
            score,
            torch.scalar_tensor(float('-inf'), device=score.device, dtype=score.dtype)
        )

        score = torch.where(
            mask.view(batch_size, 1, seq_q, seq_kv),
            self.softmax(score),
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
        )

        score = score.view(-1, seq_q, seq_kv)

        h_out = torch.bmm(
            score, h_v
        )
        h_out = h_out.view(batch_size, -1, seq_q, self.dim_head)
        h_out = h_out.permute(0, 2, 1, 3).contiguous()
        h_out = h_out.view(batch_size, seq_q, -1)
        if config['tp_size'] > 1:
            h_out = h_out.view(h_out.shape[0] * bmt.config["tp_size"], -1, h_out.shape[-1]) 

        attn_out = self.project_out(h_out)

        return attn_out
        

        


