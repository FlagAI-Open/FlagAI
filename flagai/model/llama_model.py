
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
from typing import Optional, Tuple
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from flagai.model.layers.feedforward import RowParallelLinear, ColumnParallelLinear
from flagai.model.layers.embeddings import ParallelEmbedding
import os 
from flagai.model.base_model import BaseModel 
from flagai.mpu import get_model_parallel_world_size
if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
    from flagai.mpu.utils import divide
    from flagai.mpu.random import checkpoint
    from flagai.mpu import copy_to_model_parallel_region, gather_from_model_parallel_region, get_model_parallel_world_size, get_cuda_rng_tracker
    from flagai.mpu.cross_entropy import vocab_parallel_cross_entropy

elif os.getenv('ENV_TYPE') == 'deepspeed':
    from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
else:
    from torch.utils.checkpoint import checkpoint
from flagai.model.utils import normal_init_method as n_init
normal_init_method = n_init(0,0.0001)
@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024

    use_cache: bool = True


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if os.getenv("ENV_TYPE") == 'deepspeed+mpu':
            self.n_local_heads = args.n_heads // get_model_parallel_world_size()
            self.head_dim = args.dim // args.n_heads
        else:
            self.n_local_heads = args.n_heads 
            self.head_dim = args.dim // args.n_heads
        if os.getenv("ENV_TYPE") == 'deepspeed+mpu': 
            self.wq = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=normal_init_method,
            )
            self.wk = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=normal_init_method,
            )
            self.wv = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=normal_init_method,
            )
            self.wo = RowParallelLinear(
                args.n_heads * self.head_dim,
                args.dim,
                bias=False,
                input_is_parallel=True,
                init_method=normal_init_method,
            )
        else:
            self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
            self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
            self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
            self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
            
            
        if args.use_cache:
            self.cache_k = torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
            )
            self.cache_v = torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
            )
        self.args = args

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.args.use_cache:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else :
            keys = xk
            values = xv 
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        if os.getenv("ENV_TYPE") == 'deepspeed+mpu':
            self.w1 = ColumnParallelLinear(
                dim, hidden_dim, bias=False, gather_output=False,init_method=normal_init_method
            )
            self.w2 = RowParallelLinear(
                hidden_dim, dim, bias=False, input_is_parallel=True, init_method=normal_init_method
            )
            self.w3 = ColumnParallelLinear(
                dim, hidden_dim, bias=False, gather_output=False, init_method=normal_init_method
            )
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.start_pos = 0

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), self.start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LLAMA(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
        if "use_cache" in kwargs:
            use_cache = kwargs["use_cache"]
        
        else :
            use_cache = True
        print("+"*20,kwargs)
        params: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=32,
                                        multiple_of=config["multiple_of"],
                                        dim=config["dim"],
                                        n_heads=config["n_heads"],
                                        vocab_size=config["vocab_size"],
                                        norm_eps=config["norm_eps"],
                                        n_layers=config["n_layers"],
                                        use_cache=use_cache)

        self.params = params
        if "checkpoint_activations" in kwargs:
            print("*"*20, 'init checkpointing!!!')
            self.params.checkpoint_activations = kwargs['checkpoint_activations']
        else:
            self.params.checkpoint_activations = False
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim 
        )
        from flagai.logger import log_dist
        log_dist(f"self.tok_embeddings, {self.tok_embeddings.weight}")
    
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        
        if os.getenv("ENV_TYPE") == 'deepspeed+mpu':
            self.output = ColumnParallelLinear(
                params.dim, params.vocab_size, bias=False,init_method=normal_init_method
            )
        else:
            self.output = nn.Linear(params.dim, params.vocab_size, bias = False)
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.loss_func = nn.CrossEntropyLoss()


    def forward(self, input_ids: torch.Tensor, start_pos=0, labels=None, **kwargs):
        _bsz, seqlen = input_ids.shape
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        if self.params.checkpoint_activations:
            h = checkpoint(
                create_custom_forward(self.tok_embeddings),
                input_ids
            )
        else:
            h = self.tok_embeddings(input_ids)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            layer.start_pos = start_pos 
            if self.params.checkpoint_activations:
                # print("*"*20, 'apply checkpointing!!!')
                h = checkpoint(create_custom_forward(layer), 
                               h,
                               freqs_cis, 
                               mask)

            else:
                h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        # print(f"h size -> {h.shape}") 
        if labels is not None:
            if self.params.checkpoint_activations:
                h = checkpoint(
                    create_custom_forward(self.output),
                    h
                )
            else:
                h = self.output(h)
            # print(f"output size -> {h.shape}")
            # print(f"lables size -> {labels.shape}")
            loss = self.loss_func(
                h[..., :-1, :].reshape(-1, self.params.vocab_size).contiguous().float(), labels[..., 1:].reshape(-1).contiguous().long()).mean()
           
            return {
                'logits': h,
                'loss': loss
            }
        else :
            h = self.output(h[:, -1, :])  # only compute last logits
            # return output.float()
            return {
                "logits": h.float()
            }
        

    def load_weights(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"model params are loaded successfully...")

    @classmethod
    def from_pretrain(cls,
                      download_path='./checkpoints/',
                      model_name='RoBERTa-base-ch',
                      only_download_config=False,
                      device="cpu",
                      **kwargs):
        if only_download_config:
            pretrained_model_name_or_path = os.path.join(download_path, model_name)
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            model = LLAMA.init_from_json(os.path.join(pretrained_model_name_or_path, "config.json"), **kwargs)
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            super().download(download_path, model_name)
            pretrained_model_name_or_path = os.path.join(download_path, model_name)
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            world_size = int(os.environ.get("WORLD_SIZE", -1))

            checkpoints = sorted(Path(pretrained_model_name_or_path).glob("*.pth"))
            assert (
                world_size == len(checkpoints)
            ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
            ckpt_path = checkpoints[local_rank]
            print("Loading")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            model = LLAMA.init_from_json(os.path.join(pretrained_model_name_or_path, "config.json"), **kwargs)
            torch.set_default_tensor_type(torch.FloatTensor)
            
            model.load_state_dict(checkpoint, strict=False)
        return model 