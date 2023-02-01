# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch OPT model."""
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from flagai.model.layers.activations import ACT2FN
from flagai.model.gpt2_model import GPT2Model, GPT2Stack, GPT2Config
from torch.utils.checkpoint import checkpoint

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]

class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)

class OPTStack(GPT2Stack):
    def __init__(self, config: GPT2Config):
        super(OPTStack, self).__init__(config)
        self.wpe = OPTLearnedPositionalEmbedding(config.n_positions, config.hidden_size)
        self.ln_f = None
        if config.do_layer_norm_before:
            self.ln_f = nn.LayerNorm(config.hidden_size)

        if config.n_embd != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.n_embd, bias=False)
        else:
            self.project_out = None

        if config.n_embd != config.hidden_size:
            self.project_in = nn.Linear(config.n_embd, config.hidden_size, bias=False)
        else:
            self.project_in = None

    def get_position_embeddings(self, **kwargs):
        pass
        padding_mask = kwargs["padding_mask"]
        past_length = kwargs["past_length"]
        position_embeds = self.wpe(padding_mask, past_length)
        return position_embeds

def trans_opt_to_gpt_config(opt_config_json):
    trans_config_json = {}
    trans_key = {
        "ffn_dim": "n_inner",
        "hidden_size": "hidden_size",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "vocab_size": "vocab_size",
        "activation_function": "activation_function",
        "checkpoint_activations": "checkpoint_activations",
        "word_embed_proj_dim": "n_embd",
        "do_layer_norm_before": "do_layer_norm_before",
    }
    for k, v in opt_config_json.items():
        if k in trans_key:
            trans_config_json[trans_key[k]] = v

    return trans_config_json

class OPTModel(GPT2Model):

    def __init__(self, config, **kwargs):
        config = trans_opt_to_gpt_config(config)
        super(OPTModel, self).__init__(config, **kwargs)
        self.transformer = OPTStack(self.config_gpt)

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device("cpu"))
        if "module" in checkpoint:
            # ddp
            checkpoint = checkpoint["module"]

        checkpoint_ = {}
        for k, v in checkpoint.items():
            if k[:6] == "model.":
                checkpoint_[k[6:]] = v
            else :
                checkpoint_[k] = v

        checkpoint = self.transpose_weight(checkpoint_)
        self.load_state_dict(checkpoint, strict=False)
        self.lm_head.weight.data = nn.Parameter(self.transformer.wte.weight.data)

        return checkpoint

    def transpose_weight(self, checkpoints):

        checkponts_ = {
            "transformer.wte.weight": checkpoints["decoder.embed_tokens.weight"],
            "transformer.wpe.weight": checkpoints["decoder.embed_positions.weight"],
                       }

        if "decoder.project_in.weight" in checkpoints:
            checkponts_["transformer.project_in.weight"] = checkpoints["decoder.project_in.weight"]
            checkponts_["transformer.project_out.weight"] = checkpoints["decoder.project_out.weight"]

        if "decoder.final_layer_norm.weight" in checkpoints:
            checkponts_["transformer.ln_f.weight"] = checkpoints["decoder.final_layer_norm.weight"]
            checkponts_["transformer.ln_f.bias"] = checkpoints["decoder.final_layer_norm.bias"]

        q_weight = None
        k_weight = None
        v_weight = None
        q_bias = None
        k_bias = None
        v_bias = None
        for k, v in checkpoints.items():
            # first ln
            if "decoder.layers" in k and "self_attn_layer_norm" in k:
                layer_id = k.split(".")[2]
                weight_or_bias = k.split(".")[-1]
                checkponts_[f"transformer.h.{layer_id}.ln_1.{weight_or_bias}"] = v
                continue

            # qkv
            if "self_attn.k_proj.weight" in k:
                k_weight = v
                continue
            if "self_attn.k_proj.bias" in k:
                k_bias = v
                continue

            if "self_attn.v_proj.weight" in k:
                v_weight = v
                continue
            if "self_attn.v_proj.bias" in k:
                v_bias = v
                continue

            if "self_attn.q_proj.weight" in k:
                q_weight = v
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                layer_id = k.split(".")[2]
                checkponts_[f"transformer.h.{layer_id}.attn.c_attn.weight"] = qkv_weight
                continue

            if "self_attn.q_proj.bias" in k:
                q_bias = v
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                layer_id = k.split(".")[2]
                checkponts_[f"transformer.h.{layer_id}.attn.c_attn.bias"] = qkv_bias
                continue

            # att out
            if "decoder.layers" in k and "self_attn.out_proj" in k:
                layer_id = k.split(".")[2]
                weight_or_bias = k.split(".")[-1]
                checkponts_[f"transformer.h.{layer_id}.attn.c_proj.{weight_or_bias}"] = v
                continue

            # fc1
            if "decoder.layers" in k and "fc1" in k:
                layer_id = k.split(".")[2]
                weight_or_bias = k.split(".")[-1]
                checkponts_[f"transformer.h.{layer_id}.mlp.c_fc.{weight_or_bias}"] = v
                continue

            # fc2
            if "decoder.layers" in k and "fc2" in k:
                layer_id = k.split(".")[2]
                weight_or_bias = k.split(".")[-1]
                checkponts_[f"transformer.h.{layer_id}.mlp.c_proj.{weight_or_bias}"] = v
                continue

            # second ln
            if "decoder.layers" in k and "final_layer_norm" in k:
                layer_id = k.split(".")[2]
                weight_or_bias = k.split(".")[-1]
                checkponts_[f"transformer.h.{layer_id}.ln_2.{weight_or_bias}"] = v
                continue

        return checkponts_