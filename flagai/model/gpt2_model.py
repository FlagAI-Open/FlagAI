# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import os
from flagai.model.blocks.gpt2_block import GPT2Block
from flagai.model.layers.embeddings import VocabParallelEmbedding
from flagai.model.utils import normal_init_method
from flagai.model.base_model import BaseModel

if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
    from flagai.mpu import get_model_parallel_world_size
    from flagai.mpu import gather_from_model_parallel_region
    from flagai.mpu import get_cuda_rng_tracker
    from flagai.mpu.utils import divide
if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
    from flagai.mpu import copy_to_model_parallel_region
    from flagai.mpu.random import checkpoint
elif os.getenv('ENV_TYPE') == 'deepspeed':
    from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
else:
    from torch.utils.checkpoint import checkpoint

class GPT2Config:

    def __init__(
            self,
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            gradient_checkpointing=False,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            checkpoint_activations=False,
            hidden_size=None,
            do_layer_norm_before=True,
    ):

        self.do_layer_norm_before = do_layer_norm_before
        self.checkpoint_activations = checkpoint_activations
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        if hidden_size is None:
            self.hidden_size = n_embd
        else :
            self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

class GPT2Stack(nn.Module):

    def __init__(self, config):
        self.config = config
        super().__init__()
        if os.getenv("ENV_TYPE") == "deepspeed+mpu":
            self.wte = VocabParallelEmbedding(
                config.vocab_size,
                config.n_embd,
                init_method=normal_init_method(
                    mean=0.0, std=config.initializer_range))
        else:
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
            GPT2Block(config.n_ctx, config, scale=True)
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd,
                                 eps=config.layer_norm_epsilon)
        self.device_map = None

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(0,
                                        input_shape[-1],
                                        dtype=torch.long,
                                        device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = (1.0 - attention_mask) * -10000.0

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1), )

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            if self.config.checkpoint_activations:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                outputs = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    None,
                    use_cache,
                    output_attentions,
                )
            else:

                outputs = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1], )

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1], )

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        return hidden_states


class GPT2Model(BaseModel):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        if type(config) is dict:
            config_gpt = GPT2Config(
                activation_function=self.config["activation_function"],
                n_head=self.config["n_head"],
                n_embd=self.config["n_embd"],
                n_layer=self.config["n_layer"],
                n_positions=self.config["n_positions"],
                vocab_size=self.config["vocab_size"],
                n_inner=self.config["n_inner"],
                checkpoint_activations=self.config["checkpoint_activations"],
                hidden_size=self.config.get("hidden_size", None),
                do_layer_norm_before=self.config.get("do_layer_norm_before", True),
            )
            self.config = config_gpt

        self.transformer = GPT2Stack(self.config)
        self.lm_head = nn.Linear(self.config.n_embd,
                                 self.config.vocab_size,
                                 bias=False)

    def _make_causal_mask(self, input_ids):
        device = input_ids.device
        bsz, tgt_len = input_ids.shape
        mask = torch.full((tgt_len, tgt_len), 0.0).to(device)
        mask_cond = torch.arange(mask.size(-1)).to(device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1),
                          1.0)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

    def forward(
        self,
        **data,
    ):
        input_ids = data.get("input_ids", None)
        attention_mask = data.get("attention_mask", None)
        position_ids = data.get("position_ids", None)
        labels = data.get("labels", None)
        use_cache = data.get("use_cache", None)
        output_attentions = data.get("output_attentions", None)
        output_hidden_states = data.get("output_hidden_states", None)

        extend_mask = (input_ids > 0).float()
        if attention_mask is None:
            attention_mask = self._make_causal_mask(input_ids)
            extend_mask = extend_mask.unsqueeze(1).unsqueeze(
                1) * attention_mask

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=extend_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs

        lm_logits = self.lm_head(hidden_states)

        return_data = {"logits": lm_logits}
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            return_data["loss"] = loss

        return return_data

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device("cpu"))
        if "module" in checkpoint:
            # ddp
            checkpoint = checkpoint["module"]

        checkpoint = self.transpose_weight(checkpoint)

        self.load_state_dict(checkpoint, strict=False)
        return checkpoint

    def transpose_weight(self, checkponts):
        weight_layers_same = [
            "attn.c_attn.weight", "mlp.c_fc.weight", "mlp.c_proj.weight",
            "attn.c_proj.weight"
        ]
        weight_layers_extend = []
        for layer in weight_layers_same:
            for i in range(self.config.n_layer):
                weight_layers_extend.append(f"transformer.h.{i}.{layer}")

        checkponts_ = {}
        for k, v in checkponts.items():
            if k in weight_layers_extend:
                checkponts_[k] = v.transpose(0, 1)

            else:
                checkponts_[k] = v

        return checkponts_
