# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn


def multilabel_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12
                  )  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


class GlobalPointer(nn.Module):

    def __init__(self, hidden_size, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hidden_size
        self.dense = nn.Linear(self.hidden_size,
                               self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len,
                                    dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack(
            [torch.sin(embeddings),
             torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat(
            (batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings,
                                   (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def rope(self, batch_size, seq_len, dim, qw, kw):
        # pos_emb:(batch_size, seq_len, inner_dim)
        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, dim)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        return qw, kw

    def forward(self, last_hidden_state, padding_mask):
        self.device = last_hidden_state.device

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[
            ..., self.inner_dim:]  # TODO:修改为Linear获取？

        if self.RoPE:

            qw, kw = self.rope(batch_size, seq_len, self.inner_dim, qw, kw)

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = padding_mask.unsqueeze(1).unsqueeze(1).expand(
            batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim**0.5

    def compute_loss(self, logits, labels):
        # logits:
        # labels:

        bh = logits.shape[0] * logits.shape[1]
        labels = torch.reshape(labels, shape=(bh, -1))
        logits = torch.reshape(logits, shape=(bh, -1))
        return multilabel_crossentropy(logits, labels)
