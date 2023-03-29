# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""Utilities for wrapping BertModel."""
from flagai.model.blocks.bert_block import BertBlock
from flagai.model.layers.embeddings import BertEmbeddings
from flagai.model.layers.layer_norm import BertLayerNorm
from flagai.model.layers.feedforward import BertPooler
from flagai.model.base_model import BaseModel
from flagai.model.layers.activations import ACT2FN
from flagai.model.layers.global_pointer import GlobalPointer
import torch
from flagai.model.layers.crf import CRFLayer
from torch import nn
from typing import List
import os
if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
    from flagai.mpu.random import checkpoint
elif os.getenv('ENV_TYPE') == 'deepspeed':
    from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
else:
    from torch.utils.checkpoint import checkpoint


def init_bert_weights(module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class BertStack(torch.nn.Module):

    def __init__(self, config, num_hidden_layers, hidden_size,
                 num_attention_heads, attention_probs_dropout_prob,
                 initializer_range, layernorm_epsilon, hidden_dropout_prob,
                 intermediate_size, hidden_act, enable_flash_atten=False):
        super(BertStack, self).__init__()
        self.config = config
        self.layer = torch.nn.ModuleList([
            BertBlock(hidden_size, num_attention_heads,
                      attention_probs_dropout_prob, initializer_range,
                      layernorm_epsilon, hidden_dropout_prob,
                      intermediate_size, hidden_act, enable_flash_atten)
            for _ in range(num_hidden_layers)
        ])

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True,
                **kwargs):
        all_encoder_layers = []

        for i, layer_module in enumerate(self.layer):

            if self.config['checkpoint_activations']:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = checkpoint(create_custom_forward(layer_module),
                                           hidden_states, attention_mask, **kwargs)
            else:
                hidden_states = layer_module(hidden_states, attention_mask, **kwargs)

            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        return all_encoder_layers


class BertModel(BaseModel):

    def __init__(self, config, **kwargs):

        super(BertModel, self).__init__(config, **kwargs)
        hidden_size = config["hidden_size"]
        intermediate_size = 4 * hidden_size
        self.hidden_size = hidden_size
        self.hidden_act = config["hidden_act"]
        self.vocab_size = config["vocab_size"]
        self.initializer_range = config["initializer_range"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.type_vocab_size = config["type_vocab_size"]
        self.layernorm_epsilon = config.get("layernorm_epsilon", None)
        if self.layernorm_epsilon is None:
            self.layer_norm_eps = config.get("layer_norm_eps", 0.000001)
            self.layernorm_epsilon = self.layer_norm_eps
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_probs_dropout_prob = config[
            "attention_probs_dropout_prob"]
        self.enable_flash_atten = config.get("enable_flash_atten", False)
        self.embeddings = BertEmbeddings(self.vocab_size, self.hidden_size,
                                         self.initializer_range,
                                         self.max_position_embeddings,
                                         self.type_vocab_size,
                                         self.layernorm_epsilon,
                                         self.hidden_dropout_prob)
        self.encoder = BertStack(
            config, self.num_hidden_layers, hidden_size,
            self.num_attention_heads, self.attention_probs_dropout_prob,
            self.initializer_range, self.layernorm_epsilon,
            self.hidden_dropout_prob, intermediate_size, self.hidden_act,
            self.enable_flash_atten)
        self.pooler = BertPooler(hidden_size)

        self.apply(init_bert_weights)


    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                output_all_encoded_layers=True,
                **kwargs):

        # if attention_mask is None:
        # attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = (input_ids > 0).float()
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask is not None:
            input_attention_mask_dim = len(attention_mask.shape)
            if input_attention_mask_dim == 4:
                # seq2seq mask
                extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)
            elif input_attention_mask_dim == 3:
                extended_attention_mask = extended_attention_mask.unsqueeze(1)
            elif input_attention_mask_dim == 2:
                # not need to extend
                pass
            extended_attention_mask = extended_attention_mask * attention_mask

        # extended_attention_mask need to extend to 4 dimentions.
        extended_attention_mask_dim = len(extended_attention_mask.shape)
        if extended_attention_mask_dim == 2:
            extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)
        elif extended_attention_mask_dim == 3:
            extended_attention_mask = extended_attention_mask.unsqueeze(1)
        elif extended_attention_mask_dim == 4:
            pass
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.encoder.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        kwargs['input_ids'] = input_ids
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            **kwargs)
        sequence_representation = encoded_layers[-1]
        for p in self.pooler.parameters():
            if p is None:
                continue
            sequence_representation = sequence_representation.type_as(p)
            break
        pooled_output = self.pooler(sequence_representation)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return encoded_layers, pooled_output

    def load_huggingface_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device("cpu"))
        if "module" in checkpoint:
            # ddp
            checkpoint = checkpoint["module"]
        checkpoint_new = self.convert_checkpoint_to_load(checkpoint)
        self.load_state_dict(checkpoint_new, strict=False)

        return checkpoint_new

    def load_weights(self, checkpoint_path):
        self.load_huggingface_weights(checkpoint_path)

    def convert_checkpoint_to_load(self, checkpoint):
        checkpoint_model = {}
        for k, v in checkpoint.items():
            if k[:4] == "bert":
                k_new = "model" + k[4:]
            else:
                k_new = k
            checkpoint_model[k_new] = v
        index = 0
        save_qkv_weight = []
        save_qkv_bias = []
        checkpoint_converted = {}
        for k, v in checkpoint_model.items():
            # fit bert-chinese
            if "LayerNorm.gamma" in k:
                k = k.replace("gamma", "weight")
            elif "LayerNorm.beta" in k:
                k = k.replace("beta", "bias")

            if str(index) not in k:
                checkpoint_converted[k[6:] if k[:5] == "model" else k] = v
            else:
                if "query.weight" in k or 'key.weight' in k or "value.weight" in k:
                    save_qkv_weight.append(v)

                elif "query.bias" in k or 'key.bias' in k or "value.bias" in k:
                    save_qkv_bias.append(v)
                else:
                    checkpoint_converted[k[6:] if k[:5] == "model" else k] = v
                    continue

                if "value.bias" in k:
                    checkpoint_converted[
                        f"encoder.layer.{index}.attention.self.query_key_value.weight"] = torch.cat(
                            save_qkv_weight, dim=0)
                    checkpoint_converted[
                        f"encoder.layer.{index}.attention.self.query_key_value.bias"] = torch.cat(
                            save_qkv_bias, dim=0)
                    save_qkv_weight = []
                    save_qkv_bias = []
                    index += 1

        return checkpoint_converted


class Predictions(nn.Module):

    def __init__(self, vocab_size, hidden_size, layer_nrom_eps, hidden_act):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size,
                                                     layer_nrom_eps,
                                                     hidden_act)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        # self.bias = nn.Parameter(torch.zeros(vocab_size))
        # self.decoder.bias = self.bias

    def forward(self, x):
        x = self.transform(x)
        return x, self.decoder(x)


class CLS(nn.Module):

    def __init__(self, vocab_size, hidden_size, layer_norm_eps, hidden_act):
        super().__init__()
        self.predictions = Predictions(vocab_size, hidden_size, layer_norm_eps,
                                       hidden_act)

    def forward(self, x):
        return self.predictions(x)


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, hidden_size, layer_norm_eps, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = ACT2FN[hidden_act]
        self.LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


def load_lm_layer_weight(self, checkpoints):
    self.cls.predictions.transform.dense.weight = nn.Parameter(
        checkpoints["cls.predictions.transform.dense.weight"])
    self.cls.predictions.transform.dense.bias = nn.Parameter(
        checkpoints["cls.predictions.transform.dense.bias"])

    self.cls.predictions.transform.LayerNorm.weight = nn.Parameter(
        checkpoints["cls.predictions.transform.LayerNorm.weight"])
    self.cls.predictions.transform.LayerNorm.bias = nn.Parameter(
        checkpoints["cls.predictions.transform.LayerNorm.bias"])

    self.cls.predictions.decoder.weight = nn.Parameter(
        checkpoints["cls.predictions.decoder.weight"])

    if "cls.predictions.bias" in checkpoints:
        self.cls.predictions.decoder.bias = nn.Parameter(
            checkpoints["cls.predictions.bias"])

    if "cls.predictions.decoder.bias" in checkpoints:
        self.cls.predictions.decoder.bias = nn.Parameter(
            checkpoints["cls.predictions.decoder.bias"])


def load_extend_layer_weight(self, checkpoints, extend_layer: List[str]):
    checkpoints_save = {}
    for layer in extend_layer:
        if layer in checkpoints:
            checkpoints_save[layer] = checkpoints[layer]
    self.load_state_dict(checkpoints_save, strict=False)


class BertForSeq2seq(BaseModel):

    def __init__(self, config, **kwargs):
        super(BertForSeq2seq, self).__init__(config, **kwargs)
        self.model = BertModel(config)
        self.cls = CLS(self.model.vocab_size, self.model.hidden_size,
                       self.model.layernorm_epsilon, self.model.hidden_act)

    def compute_loss(self, **data):

        pred = data["logits"]
        label = data["labels"]
        token_type_ids = data["segment_ids"]
        pred = pred[:, :-1].contiguous()
        target_mask = token_type_ids[:, 1:].contiguous()
        target_mask = target_mask.view(-1).float()

        pred = pred.view(-1, self.model.vocab_size)
        label = label.view(-1)
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        return (loss(pred, label) * target_mask
                ).sum() / target_mask.sum()  ## Deinfluence pad and prediction of sentenceA through mask

    def make_unilm_mask(self, token_type_id, seq_len):
        device = token_type_id.device

        ones = torch.ones((1, 1, seq_len, seq_len),
                          dtype=torch.float32,
                          device=device)
        a_mask = ones.tril()
        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask
        return a_mask

    def forward(self, **data):
        input_ids = data["input_ids"]
        token_type_ids = data["segment_ids"]
        labels = data.get("labels", None)

        return_data = {}
        input_shape = input_ids.shape
        seq_len = input_shape[1]
        a_mask = self.make_unilm_mask(token_type_ids, seq_len)
        encoder_out, pooler_out = self.model(
            token_type_ids=token_type_ids,
            attention_mask=a_mask,
            output_all_encoded_layers=True,
            **data,
        )
        '''
        encoder_out, pooler_out = self.model(
            input_ids,
            token_type_ids,
            a_mask,
            True,
            **data
        )
        '''
        sequence_out, decoder_out = self.cls(encoder_out[-1])

        return_data["logits"] = decoder_out
        return_data["hidden_states"] = sequence_out
        if labels is not None:
            return_data["loss"] = self.compute_loss(
                **{
                    "logits": decoder_out,
                    "labels": labels,
                    "segment_ids": token_type_ids
                })

        return return_data

    def load_weights(self, model_path):
        checkpoints = self.model.load_huggingface_weights(model_path)
        load_lm_layer_weight(self, checkpoints)


class BertForMaskLM(BaseModel):

    def __init__(self, config, **kwargs):
        super(BertForMaskLM, self).__init__(config, **kwargs)
        self.model = BertModel(config)
        self.cls = CLS(self.model.vocab_size, self.model.hidden_size,
                       self.model.layer_norm_eps, self.model.hidden_act)

    def forward(self, **data):

        input_ids = data["input_ids"]
        token_type_ids = data["segment_ids"]
        labels = data.get("labels", None)
        attention_mask = data.get("attention_mask", None)

        return_data = {}
        encoder_out, pooler_out = self.model(
            input_ids,
            token_type_ids,
            attention_mask,
            True,
        )
        sequence_out, decoder_out = self.cls(encoder_out[-1])

        return_data["logits"] = decoder_out
        return_data["hidden_states"] = sequence_out

        if labels is not None:
            return_data["loss"] = self.compute_loss(**{
                "logits": decoder_out,
                "labels": labels
            })

        return return_data

    def compute_loss(self, **data):
        pred = data["logits"]
        label = data["labels"]
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(pred.view(-1, self.config["vocab_size"]),
                         label.view(-1))
        return loss

    def load_weights(self, model_path):
        checkpoints = self.model.load_huggingface_weights(model_path)
        load_lm_layer_weight(self, checkpoints)


class BertForClsClassifier(BaseModel):

    def __init__(self, config, **kwargs):
        super(BertForClsClassifier, self).__init__(config, **kwargs)
        assert config['class_num'] != -1 and config['class_num'] is not None
        self.target_size = config['class_num']
        self.model = BertModel(config)
        self.out_layer = nn.Linear(config["hidden_size"], self.target_size)

    def forward(self, **data):

        input_ids = data["input_ids"]
        token_type_ids = data["segment_ids"]
        labels = data.get("labels", None)

        return_data = {}
        sequence_out, pooler_out = self.model(input_ids, token_type_ids, None,
                                              True)
        out = self.out_layer(pooler_out)
        return_data["logits"] = out
        return_data["hidden_states"] = sequence_out

        if labels is not None:
            return_data["loss"] = self.compute_loss(**{
                "logits": out,
                "labels": labels
            })

        return return_data

    def compute_loss(self, **data):
        pred = data["logits"]
        label = data["labels"]
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(pred.view(-1, self.target_size), label.view(-1))
        return loss

    def load_weights(self, checkpoint_path):
        checkpoints = self.model.load_huggingface_weights(checkpoint_path)
        load_extend_layer_weight(
            self,
            checkpoints,
            extend_layer=["out_layer.weight", "out_layer.bias"])


class BertForSequenceLabeling(BaseModel):

    def __init__(self, config, **kwargs):
        super(BertForSequenceLabeling, self).__init__(config, **kwargs)
        self.model = BertModel(config)
        self.final_dense = nn.Linear(self.model.hidden_size,
                                     config['class_num'])

    def compute_loss(self, logits, labels, target_mask):
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = (loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1)) *
                target_mask.view(-1)).sum() / target_mask.sum()
        return loss

    def forward(self, **data):

        input_ids = data["input_ids"]
        token_type_ids = data.get("segment_ids", None)
        labels = data.get("labels", None)
        padding_mask = (input_ids > 0).float()

        return_data = {}
        sequence_out, pooler_out = self.model(input_ids, token_type_ids, None,
                                              True)
        sequence_out = sequence_out[-1]

        logits = self.final_dense(sequence_out)

        return_data["logits"] = logits
        return_data["hidden_states"] = sequence_out

        if labels is not None:
            return_data["loss"] = self.compute_loss(logits, labels,
                                                    padding_mask)

        return return_data

    def load_weights(self, model_path):
        checkpoints = self.model.load_huggingface_weights(model_path)
        load_extend_layer_weight(
            self,
            checkpoints,
            extend_layer=["final_dense.weight", "final_dense.bias"])


class BertForSequenceLabelingCRF(BaseModel):
    """
    """

    def __init__(self, config, **kwargs):
        super(BertForSequenceLabelingCRF, self).__init__(config, **kwargs)
        self.model = BertModel(config)
        self.final_dense = nn.Linear(self.model.hidden_size,
                                     config['class_num'])
        self.crf_layer = CRFLayer(config['class_num'])

    def compute_loss(self, logits, labels, target_mask):
        loss = self.crf_layer(logits, labels, target_mask)
        return loss.mean()

    def forward(self, **data):

        input_ids = data["input_ids"]
        token_type_ids = data.get("segment_ids", None)
        labels = data.get("labels", None)
        padding_mask = (input_ids > 0).float()

        return_data = {}
        sequence_out, pooler_out = self.model(input_ids, token_type_ids, None,
                                              True)
        sequence_out = sequence_out[-1]

        logits = self.final_dense(sequence_out)

        return_data["logits"] = logits
        return_data["hidden_states"] = sequence_out

        if labels is not None:
            return_data["loss"] = self.compute_loss(logits, labels,
                                                    padding_mask)

        return return_data

    def load_weights(self, model_path):
        checkpoints = self.model.load_huggingface_weights(model_path)
        load_extend_layer_weight(self,
                                 checkpoints,
                                 extend_layer=[
                                     "final_dense.weight", "final_dense.bias",
                                     "crf_layer.trans"
                                 ])


class BertForSequenceLabelingGP(BaseModel):
    """
    """

    def __init__(self, config, **kwargs):
        super(BertForSequenceLabelingGP, self).__init__(config, **kwargs)
        self.model = BertModel(config)
        self.gp = GlobalPointer(self.model.hidden_size,
                                config['class_num'],
                                config['inner_dim'],
                                RoPE=True)

    def forward(self, **data):
        input_ids = data["input_ids"]
        token_type_ids = data.get("segment_ids", None)
        padding_mask = (input_ids > 0).float()
        labels = data.get("labels", None)
        sequence_out, pooler_out = self.model(input_ids, token_type_ids, None,
                                              True)
        sequence_out = sequence_out[-1]

        gp_out = self.gp(sequence_out, padding_mask)
        return_data = {"logits": gp_out, "hidden_states": sequence_out}

        if labels is not None:
            return_data["loss"] = self.gp.compute_loss(gp_out, labels)
        return return_data

    def load_weights(self, model_path):
        checkpoints = self.model.load_huggingface_weights(model_path)
        load_extend_layer_weight(
            self,
            checkpoints,
            extend_layer=["gp.dense.weight", "gp.dense.bias"])


class BertForEmbedding(BaseModel):

    def __init__(self, config, **kwargs):
        super(BertForEmbedding, self).__init__(config, **kwargs)
        self.model = BertModel(config)

    def forward(self, **data):

        input_ids = data["input_ids"]
        token_type_ids = data["segment_ids"]

        return_data = {}
        sequence_out, pooler_out = self.model(input_ids, token_type_ids, None,
                                              True)
        out = sequence_out[-1]
        return_data["logits"] = out
        return_data["hidden_states"] = sequence_out

        return return_data

    def load_weights(self, model_path):
        self.model.load_huggingface_weights(model_path)
