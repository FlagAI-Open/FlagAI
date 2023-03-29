# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import numpy as np
import torch
import torch.nn.functional as F


def padding(indice, max_length, pad_idx=0):

    pad_indice = [
        item + [pad_idx] * max(0, max_length - len(item)) for item in indice
    ]
    return torch.tensor(pad_indice)


def bert_sequence_label_gp_collate_fn(batch):
    """
    Dynamic padding
    """

    def sequence_padding(inputs,
                         length=None,
                         value=0,
                         seq_dims=1,
                         mode='post'):
        """ padding sequence to the same lenght"""
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError(
                        '"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    token_ids = [data["input_ids"] for data in batch]
    labels = [data["labels"] for data in batch]
    token_ids_padded = sequence_padding(token_ids)
    labels_padded = sequence_padding(labels, seq_dims=3)
    token_ids_padded = torch.from_numpy(token_ids_padded)
    labels_padded = torch.from_numpy(labels_padded)

    return {"input_ids": token_ids_padded, "labels": labels_padded}


def seq2seq_collate_fn(batch):
    # bert seq2seq task collate fn
    token_ids = [data["input_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["segment_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    data = {
        "input_ids": token_ids_padded,
        "segment_ids": token_type_ids_padded,
        "labels": target_ids_padded
    }

    # TODO
    flash_atten = all([data["flash_atten"] for data in batch])
    if flash_atten:
        attention_mask = token_ids_padded > 0
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
        data['cu_seqlens'] = cu_seqlens

    return data


def bert_cls_collate_fn(batch):
    # bert cls task collate fn
    token_ids = [data["input_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["segment_ids"] for data in batch]
    target_ids = [data["labels"] for data in batch]
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)

    return {
        "input_ids": token_ids_padded,
        "segment_ids": token_type_ids_padded,
        "labels": target_ids
    }


def bert_sequence_label_collate_fn(batch):
    """
    dynamical padding
    """

    token_ids = [data["input_ids"] for data in batch]

    max_length = max([len(t) for t in token_ids])
    target_ids = [data["labels"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)

    target_ids_padded = padding(target_ids, max_length)

    return {"input_ids": token_ids_padded, "labels": target_ids_padded}
