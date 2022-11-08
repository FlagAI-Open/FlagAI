import torch
from flagai.model.blocks.cpm_block import CPM3Block
from flagai.model.layers.layer_norm import CPM3LayerNorm
from flagai.model.layers.embeddings import CPM3Embedding
from flagai.model.layers.embeddings import CPM3SegmentPositionEmbedding
from flagai.model.layers.linear import CPM3Linear

from flagai.model.base_model import BaseModel


import json
import os
import copy
from typing import Any, Dict, Union

class CPM3Stack(torch.nn.Module):
    """ Layers of encoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): number of layers.
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        att_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.Attention`. Defaults to 0.0.
        att_init_std (float, optional): init_std used in :py:class:`model_center.layer.Attention`. Defaults to 0.02.
        att_bias (bool, optional): bias used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        att_mask_value (float, optional): mask_value used in in :py:class:`model_center.layer.Attention`. Defaults to float("-inf").
        ffn_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.0.
        ffn_init_std (float, optional): init_std used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.02.
        ffn_bias (bool, optional): bias used in :py:class:`model_center.layer.FeedForward`. Defaults to False.
        ffn_activate_fn (str, optional): activate_fn used in :py:class:`model_center.layer.FeedForward`. Defaults to "gated_gelu".
        pos_bias_type (str, optional): pos_bias_type used in :py:class:`model_center.layer.Attention`. Defaults to "none".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        attn_scale (bool, optional): attn_scale used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        dropout_p (float, optional): Defaults to 0.
    """
    def __init__(self, config
        ):

        super().__init__()
        
        self.num_layers = config.num_layers

        self.layers = torch.nn.ModuleList([
            CPM3Block(config)
            for _ in range(self.num_layers)
        ])

        self.output_layernorm = CPM3LayerNorm(
                    dim_norm = config.dim_model, 
                    bias = config.norm_bias, 
                    dtype = config.dtype,
                    eps = config.norm_eps,
                    init_var = config.norm_init_var)

    def forward(self, hidden_states : torch.Tensor,
                      attention_mask : torch.Tensor,
                      position_bias : torch.Tensor = None,
                      past_key_values = None,
                      ):
        """
        Args:
            hidden-states (:obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``): Input of encoder, might be the embedding of a batch of sequences. 
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_enc, seq_enc)``): Avoid invalid areas to participate in the calculation 
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_enc, seq_enc)``) Provides position information to attention mechanism.  

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``: The encoder output. 

        """
        # (batch, seq_enc, dim_model)
        
        present_key_values = []
        for i, module in enumerate(self.layers):
            hidden_states, present_key_value = module(hidden_states, attention_mask, position_bias, None, None, None, past_key_values[i])
            present_key_values.append(present_key_value)
        # (batch, seq_enc, dim_model)
        hidden_states = self.output_layernorm(hidden_states)
        return hidden_states, present_key_values


class CPM3(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        if type(config) is dict:
            config_cpm3 = CPM3Config(
                vocab_size = self.config['vocab_size'],
                        dim_model = self.config['dim_model'],
                        num_heads = self.config['num_heads'],
                        dim_head = self.config['dim_head'],
                        dim_ff = self.config['dim_ff'],
                        num_layers = self.config['num_layers'],
                        dropout_p = self.config['dropout_p'],
                        emb_init_mean = self.config['emb_init_mean'],
                        emb_init_std = self.config['emb_init_std'],
                        pos_bias_type = self.config['pos_bias_type'],
                        position_bias_num_buckets = self.config['position_bias_num_buckets'],
                        position_bias_max_distance = self.config['position_bias_max_distance'],
                        pos_init_mean = self.config['pos_init_mean'],
                        pos_init_std = self.config['pos_init_std'],
                        norm_init_var = self.config['norm_init_var'],
                        norm_bias = self.config['norm_bias'],
                        norm_eps = self.config['norm_eps'],
                        att_init_mean = self.config['att_init_mean'], 
                        att_init_std = self.config['att_init_std'],
                        att_bias = self.config['att_bias'],
                        att_mask_value = float(self.config['att_mask_value']) if type(self.config['att_mask_value']) == str else self.config['att_mask_value'],
                        ffn_init_mean = self.config['ffn_init_mean'], 
                        ffn_init_std = self.config['ffn_init_std'],
                        ffn_bias = self.config['ffn_bias'],
                        ffn_activate_fn = self.config['ffn_activate_fn'],
                        proj_init_mean = self.config['proj_init_mean'],
                        proj_init_std = self.config['proj_init_std'],
                        proj_bias = self.config['proj_bias'],
                        length_scale = self.config['length_scale'],
                        attn_scale = self.config['attn_scale'],
                        half = self.config['half'], 
                        int8 = self.config['int8'],
                        tied = self.config['tied'],
                        prompt_types = self.config['prompt_types'],
                        prompt_length = self.config['prompt_length'], 
                        segment_types = self.config['segment_types'],
                        max_exact_rate = self.config['max_exact_rate'],
                        max_distance_rate = self.config['max_distance_rate'],
                        absolute_inner_segment = self.config['absolute_inner_segment'],
                        cls_head = self.config.get('cls_head', None),
                        post_layer_norm= self.config.get('post_layer_norm', None)
            )
        self.config = config_cpm3
        
        self.encoder = CPM3Stack(self.config)
        self.prompt_embedding = CPM3Embedding(
            vocab_size = self.config.prompt_types * self.config.prompt_length, 
            embedding_size = self.config.dim_model,
            length_scale = self.config.length_scale,
            dtype = self.config.dtype,
            int8 = self.config.int8,)

        self.input_embedding = CPM3Embedding(
            vocab_size = self.config.vocab_size, 
            embedding_size = self.config.dim_model,
            length_scale = self.config.length_scale,
            dtype = self.config.dtype,
            int8 = self.config.int8,)

        self.position_bias = CPM3SegmentPositionEmbedding(
            num_segments = self.config.segment_types,
            num_heads = self.config.num_heads, 
            num_buckets = self.config.position_bias_num_buckets, 
            max_distance = self.config.position_bias_max_distance, 
            max_exact_rate = self.config.max_exact_rate,
            max_distance_rate = self.config.max_distance_rate,
            absolute_inner_segment = self.config.absolute_inner_segment,
            bidirectional = True,
            dtype = self.config.dtype,)
        
        self.prompt_length = self.config.prompt_length
        self.tied = self.config.tied
        self.cls_head = self.config.cls_head
        if self.cls_head:
            self.output_projection = CPM3Linear(
                vocab_size = self.config.cls_head,
                embedding_size = self.config.dim_model,
                length_scale = self.config.length_scale,
                dtype = self.config.dtype,
                int8 = self.config.int8,
                init_mean = self.config.proj_init_mean,
                init_std = self.config.proj_init_std,
                bias = self.config.proj_bias,)
        elif not self.config.tied:
            self.output_projection = CPM3Linear(
                vocab_size = self.config.vocab_size,
                embedding_size = self.config.dim_model,
                length_scale = self.config.length_scale,
                dtype = self.config.dtype,
                int8 = self.config.int8,
                init_mean = self.config.proj_init_mean,
                init_std = self.config.proj_init_std,
                bias = self.config.proj_bias,)

    def forward(self, input : torch.Tensor, # (batch, seqlen)
                      length : torch.Tensor, # (batch)
                      context : torch.Tensor, # (batch, seqlen)
                      position: torch.Tensor, # (batch, seqlen)
                      segment: torch.Tensor, # (batch, seqlen)
                      span : torch.Tensor,  # (batch, seqlen)
                      past_key_values = None,  # num_layers * 2 * (batch, num_heads, seqlen, dim_head)
                      right_ctx_start_idx = None,
                      last_input_idx = None,
                      cached_attn_mask_pos_bias = None
                    ):

        batch = input.size(0)

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.encoder.num_layers)

            input_prompt = input[:, :self.prompt_length].contiguous()
            input_ids = input[:, self.prompt_length:].contiguous()

            prompt_states = self.prompt_embedding(input_prompt)
            hidden_states = self.input_embedding(input_ids)
            hidden_states = torch.cat([prompt_states, hidden_states], 1)
            
            input_seqlen = input.size(1) - (right_ctx_start_idx - last_input_idx) + 1
        else:
            past_length = past_key_values[0][0].size(-2)
            hidden_states = self.input_embedding(input)
            
            input_seqlen = input.size(1) # input_seqlen = 1

        valid_len = input_seqlen + past_length

        if cached_attn_mask_pos_bias is None:
            with torch.no_grad():
                device = input.device
                seqlen = context.size(1)
                directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(-1, 1)
                attention_mask = context[:, None, :] | (context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen))
                attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
                mask_1d = torch.arange(seqlen, device=device)[None, :].repeat(batch, 1) < length[:, None]
                attention_mask = mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask

            position_bias = self.position_bias(position, position, segment, segment)

            assert attention_mask.size(1) == attention_mask.size(2) == position_bias.size(2) == position_bias.size(3)
            switch_indices = [x for x in range(right_ctx_start_idx, attention_mask.size(1))] + [x for x in range(right_ctx_start_idx)]
            attention_mask = attention_mask[:, switch_indices, :]
            attention_mask = attention_mask[:, :, switch_indices]
            position_bias = position_bias[:, :, switch_indices, :]
            position_bias = position_bias[:, :, :, switch_indices]

            cached_attn_mask_pos_bias = (attention_mask, position_bias)
        else:
            attention_mask, position_bias = cached_attn_mask_pos_bias

        if past_length == 0:
            assert hidden_states.size(1) == attention_mask.size(1)
            hidden_states = hidden_states[:, switch_indices, :]
            hidden_states = hidden_states[:, :valid_len, :]

            attention_mask = attention_mask[:, :valid_len, :valid_len]
            position_bias = position_bias[:, :, :valid_len, :valid_len]
        else:
            attention_mask = attention_mask[:, past_length:valid_len, :valid_len]
            position_bias = position_bias[:, :, past_length:valid_len, :valid_len]

        hidden_states, present_key_values = self.encoder(hidden_states, attention_mask, position_bias, past_key_values)

        if self.cls_head:
            logits = self.output_projection(hidden_states)
        elif not self.tied:
            logits = self.output_projection(hidden_states)
        else:
            logits = self.input_embedding.projection(hidden_states)

        return logits, hidden_states, present_key_values, cached_attn_mask_pos_bias
    
    def load_weights(self, checkpoint_path):
        self.load_state_dict(
            torch.load(checkpoint_path),
            strict = True
        )


class Config(object):

    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike]):
        return cls.from_json_file(pretrained_model_name_or_path)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output


class CPM3Config(Config):

    def __init__(self, vocab_size = 30720,
                        dim_model = 4096,
                        num_heads = 64,
                        dim_head = 64,
                        dim_ff = 10240,
                        num_layers = 32,
                        dropout_p = 0.0,
                        emb_init_mean = 0.0,
                        emb_init_std = 1.0,
                        pos_bias_type = "relative",
                        position_bias_num_buckets = 512,
                        position_bias_max_distance = 2048,
                        pos_init_mean = 0.0,
                        pos_init_std = 1.0,
                        norm_init_var = 1.0,
                        norm_bias = False,
                        norm_eps = 1e-6,
                        att_init_mean = 0.0, 
                        att_init_std = 1.0,
                        att_bias = False,
                        att_mask_value = float("-inf"),
                        ffn_init_mean = 0.0, 
                        ffn_init_std = 1.0,
                        ffn_bias = False,
                        ffn_activate_fn = "gated_gelu",
                        proj_init_mean = 0.0,
                        proj_init_std = 1.0,
                        proj_bias = False,
                        length_scale = True,
                        attn_scale = True,
                        half = True, 
                        int8 = False,
                        tied = True,
                        prompt_types = 32,
                        prompt_length = 64, 
                        segment_types = 34,
                        max_exact_rate = 0.25,
                        max_distance_rate = 1.0,
                        absolute_inner_segment = True,
                        cls_head = None,
                        post_layer_norm=False,
                        is_decoder : bool = False,
                        parallel_ffn : bool = False,):

        super().__init__()
        self.prompt_types = prompt_types
        self.prompt_length = prompt_length
        self.segment_types = segment_types
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.norm_eps = norm_eps
        self.norm_init_var = norm_init_var
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.length_scale = length_scale
        self.absolute_inner_segment = absolute_inner_segment
        self.max_distance_rate = max_distance_rate
        self.max_exact_rate = max_exact_rate
        self.int8 = int8
        self.tied = tied
        if half: 
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.vocab_size = vocab_size
        self.pos_bias_type = pos_bias_type
        self.pos_init_mean = pos_init_mean
        self.pos_init_std = pos_init_std
        self.norm_bias = norm_bias
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.attn_scale = attn_scale
        self.post_layer_norm = post_layer_norm
        self.is_decoder = is_decoder
        self.parallel_ffn  = parallel_ffn