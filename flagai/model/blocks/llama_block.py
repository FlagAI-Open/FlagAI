# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.model.layers.attentions import LLAMAAttention
from flagai.model.layers.feedforward import LLAMAForward
from torch import nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LLAMABlock(nn.Module):
    def __init__(self, layer_id, config ):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = LLAMAAttention(config)

        self.feed_forward = LLAMAForward(
            dim=config.dim, hidden_dim=4 * config.dim, multiple_of=config.multiple_of, config=config
        )

        self.layer_id = layer_id
        if config.flash_atten_llama_style:
            import flash_attn
            self.attention_norm = flash_attn.ops.rms_norm.RMSNorm(config.dim, eps=config.norm_eps)
            self.ffn_norm = flash_attn.ops.rms_norm.RMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
            self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.start_pos = 0
        self.use_cache = False
    def forward(self, x, 
                freqs_cis,
                mask ):
        h = x + self.attention.forward(self.attention_norm(x), self.start_pos, freqs_cis, mask, self.use_cache)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

