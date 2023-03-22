# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
from packaging import version
from torch import Tensor, nn


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
) :
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk) 

class LLAMAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if os.getenv("ENV_TYPE") == 'deepspeed+mpu':
            self.n_local_heads = config.n_heads // get_model_parallel_world_size()
            self.head_dim = config.dim // config.n_heads
            self.wq = ColumnParallelLinear(
                config.dim,
                config.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wk = ColumnParallelLinear(
                config.dim,
                config.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wv = ColumnParallelLinear(
                config.dim,
                config.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wo = RowParallelLinear(
                config.n_heads * self.head_dim,
                config.dim,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x,
            )
        else:
            self.n_local_heads = config.n_heads 
            self.head_dim = config.dim // config.n_heads
            self.wq = nn.Linear(config.dim, config.dim, bias=False)
            self.wk = nn.Linear(config.dim, config.dim, bias=False)
            self.wv = nn.Linear(config.dim, config.dim, bias=False)
            self.wo = nn.Linear(config.dim, config.dim, bias=False)
        if config.use_cache:
            self.cache_k = torch.zeros(
                (config.max_batch_size, config.max_seq_len, self.n_local_heads, self.head_dim)
            )
            self.cache_v = torch.zeros(
                (config.max_batch_size, config.max_seq_len, self.n_local_heads, self.head_dim)
            )
        self.config = config 

    def forward(
                self, 
                x,
                start_pos,
                freqs_cis, 
                mask,
                use_cache=False,
                **kwargs):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if use_cache:
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

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) *
            (input + 0.044715 * torch.pow(input, 3.0))))


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if version.parse(
                torch.__version__) < version.parse("1.4") or use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 +
                              torch.tanh(input * 0.7978845608 *
                                         (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.
    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(
                f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)


class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.7"):
            self.act = self._silu_python
        else:
            self.act = nn.functional.silu

    def _silu_python(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(input)

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class MishActivation(nn.Module):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9"):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


ACT2FN = {
    "relu": nn.ReLU(),
    "gelu_impl": gelu_impl,
    "silu": SiLUActivation(),
    "swish": SiLUActivation(),
    "gelu": GELUActivation(),
    "tanh": nn.Tanh(),
    "gelu_python": GELUActivation(use_gelu_python=True),
    "gelu_new": NewGELUActivation(),
    "gelu_fast": FastGELUActivation(),
    "quick_gelu": QuickGELUActivation(),
    "gelu_10": ClippedGELUActivation(-10, 10),
    "mish": MishActivation(),
    "linear": LinearActivation(),
    "sigmoid": nn.Sigmoid(),
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}"
        )


# For backwards compatibility with: from activations import gelu_python
gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")
gelu_impl = get_activation("gelu_impl")
relu = get_activation("relu")
