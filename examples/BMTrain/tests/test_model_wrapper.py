from utils import *

from typing import Optional
import torch
import math
import torch.nn.functional as F
import bmtrain as bmt
import time

class Attention(torch.nn.Module):
    def __init__(self, 
            dim_model : int, dim_head : int,
            num_heads : int, bias : bool = True,
            dtype = None
        ) -> None:
        super().__init__()

        self.project_q = torch.nn.Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
        self.project_k = torch.nn.Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
        self.project_v = torch.nn.Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)

        self.project_out = torch.nn.Linear(dim_head * num_heads, dim_model, bias=bias, dtype=dtype)

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
        batch_size, seq_q, dim_model = hidden_q.size()
        seq_kv = hidden_kv.size(1)

        h_q : torch.Tensor = self.project_q(hidden_q)
        h_k : torch.Tensor = self.project_k(hidden_kv)
        h_v : torch.Tensor = self.project_v(hidden_kv)

        h_q = h_q.view(batch_size, seq_q, self.num_heads, self.dim_head)
        h_k = h_k.view(batch_size, seq_kv, self.num_heads, self.dim_head)
        h_v = h_v.view(batch_size, seq_kv, self.num_heads, self.dim_head)

        h_q = h_q.permute(0, 2, 1, 3).contiguous()
        h_k = h_k.permute(0, 2, 1, 3).contiguous()
        h_v = h_v.permute(0, 2, 1, 3).contiguous()

        h_q = h_q.view(batch_size * self.num_heads, seq_q, self.dim_head)
        h_k = h_k.view(batch_size * self.num_heads, seq_kv, self.dim_head)
        h_v = h_v.view(batch_size * self.num_heads, seq_kv, self.dim_head)

        score = torch.bmm(
            h_q, h_k.transpose(1, 2)
        )
        score = score / math.sqrt(self.dim_head)

        score = score.view(batch_size, self.num_heads, seq_q, seq_kv)

        if position_bias is not None:
            score = score + position_bias.view(batch_size, self.num_heads, seq_q, seq_kv)
        
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

        score = score.view(batch_size * self.num_heads, seq_q, seq_kv)

        h_out = torch.bmm(
            score, h_v
        )
        h_out = h_out.view(batch_size, self.num_heads, seq_q, self.dim_head)
        h_out = h_out.permute(0, 2, 1, 3).contiguous()
        h_out = h_out.view(batch_size, seq_q, self.num_heads * self.dim_head)

        attn_out = self.project_out(h_out)
        return attn_out
        
class Feedforward(torch.nn.Module):
    def __init__(self, dim_model : int, dim_ff : int, bias : bool = True, dtype = None) -> None:
        super().__init__()

        self.w_in = torch.nn.Linear(dim_model, dim_ff, bias = bias, dtype=dtype)
        self.w_out = torch.nn.Linear(dim_ff, dim_model, bias = bias, dtype=dtype)

        self.relu = torch.nn.ReLU()
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        return self.w_out(self.relu(self.w_in(input)))


class TransformerEncoder(torch.nn.Module):
    def __init__(self,
            dim_model : int, dim_head : int, num_heads : int, dim_ff : int,
            bias : bool = True, dtype = None
        ) -> None:
        super().__init__()

        self.ln_attn = torch.nn.LayerNorm(dim_model, dtype=dtype)
        self.attn = Attention(dim_model, dim_head, num_heads, bias=bias, dtype=dtype)

        self.ln_ff = torch.nn.LayerNorm(dim_model, dtype=dtype)
        self.ff = Feedforward(dim_model, dim_ff, bias=bias, dtype=dtype)
    
    def forward(self,
            hidden : torch.Tensor,      # (batch, seq_len, dim_model)
            mask : torch.BoolTensor,    # (batch, seq_len, dim_model)
            position_bias : Optional[torch.Tensor] = None,   # (batch, num_head, seq_len, seq_len)
        ):
        x = self.ln_attn(hidden)
        x = self.attn(x, x, mask, position_bias)
        hidden = hidden + x

        x = self.ln_ff(hidden)
        x = self.ff(x)
        hidden = hidden + x

        return hidden
    

class GPT(torch.nn.Module):
    def __init__(self,
            num_layers : int, vocab_size : int,
            dim_model : int, dim_head : int, num_heads : int, dim_ff : int,
            max_distance : int,
            bias : bool = True, dtype = None
        ) -> None:
        super().__init__()

        self.max_distance = max_distance

        self.word_emb = torch.nn.Embedding(vocab_size, dim_model, dtype=dtype)
        self.pos_emb = torch.nn.Embedding(max_distance, dim_model, dtype=dtype)
        self.dim_model = dim_model
        
        self.transformers = torch.nn.ModuleList([
            TransformerEncoder(
                dim_model, dim_head, num_heads, dim_ff, bias, dtype
            )
            for _ in range(num_layers)
        ])

        self.layernorm = torch.nn.LayerNorm(dim_model, dtype=dtype)

    def forward(self,
            input : torch.LongTensor,   # (batch, seq_len)
            pos : torch.LongTensor,     # (batch, seq_len)
            mask : torch.BoolTensor,    # (batch, seq_len)
        ) -> torch.Tensor:

        mask_2d = mask[:, None, :] & mask[:, :, None]   # (batch, seq_len, seq_len)
        mask_2d = mask_2d & (pos[:, None, :] >= pos[:, :, None])

        input_emb = self.pos_emb(pos) + self.word_emb(input)

        out = input_emb
        for layer in self.transformers:
            out = layer(out, mask_2d)
        out = self.layernorm(out)

        logits = F.linear(out, self.word_emb.weight) / math.sqrt(self.dim_model)

        return logits

def test_main():
    model = GPT(
        num_layers=8,
        vocab_size=10240, 
        dim_model=2560,
        dim_head=80,
        num_heads=32,
        dim_ff=8192,
        max_distance=1024,
        bias=True,
        dtype=torch.half
    )

    bmt_model = bmt.BMTrainModelWrapper(model)
    model = model.cuda()

    # use the break if i == bmt.rank() to generate different data on different rank
    torch.manual_seed(1234)
    batch_size = 2
    seq_len = 512

    for i in range(bmt.world_size()):
        sent = torch.randint(0, 10240, (batch_size, seq_len + 1))
        enc_length = torch.randint(128, seq_len, (batch_size,)).long().cuda()
        enc_input = sent[:, :-1].long().cuda()
        targets = sent[:, 1:].long().cuda()
        mask = torch.arange(seq_len).long().cuda()[None, :] < enc_length[:, None]
        targets = torch.where(
            mask,
            targets,
            torch.full_like(targets, -100, dtype=torch.long)
        )

        if i == bmt.rank():
            break

    pos = torch.arange(enc_input.size(1)).long().cuda().repeat(enc_input.size(0), 1)
    logits = model(enc_input, pos, pos < enc_length[:, None])
    bmt_logits = bmt_model(enc_input, pos, pos < enc_length[:, None])
    print(logits)
    bmt.synchronize()
    print(bmt_logits)
    assert_all_eq(logits, bmt_logits)

if __name__ == '__main__':
    bmt.init_distributed(seed=0)

    test_main()
