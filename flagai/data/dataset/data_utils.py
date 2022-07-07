# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
""" Tasks data utility."""
import copy
import json
import pickle
import re
from typing import Dict, List, Optional
import numpy as np
import random
import torch
from flagai import mpu
try:
    import deepspeed
except:
    pass
import os


def clean_text(text):
    """Remove new lines and multiple spaces and adjust end of sentence dot."""

    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    for _ in range(3):
        text = text.replace(' . ', '. ')

    return text


class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self,
                 guid,
                 text_a,
                 text_b=None,
                 label=None,
                 logits=None,
                 meta: Optional[Dict] = None,
                 idx=-1,
                 num_choices=1):
        """
        Create a new InputExample.

        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits
        self.idx = idx
        self.num_choices = num_choices
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


def num_special_tokens_to_add(text_a_ids,
                              text_b_ids,
                              answer_ids,
                              add_cls,
                              add_sep,
                              add_piece,
                              add_eos=True):
    # number of special tokens to add
    num_tokens = 0
    if add_cls:
        num_tokens += 1
    if text_b_ids and add_sep:
        num_tokens += 1
    if add_eos:
        num_tokens += 1
    if not answer_ids and add_piece:
        num_tokens += 1
    return num_tokens


def build_input_from_ids(text_a_ids,
                         text_b_ids,
                         answer_ids,
                         max_seq_length,
                         tokenizer,
                         args=None,
                         add_cls=True,
                         add_sep=False,
                         add_piece=False,
                         add_eos=True,
                         mask_id=None):

    # Prepare ids for special tokens
    if mask_id is None:
        mask_id = tokenizer.get_command_id('MASK')
    eos_id = tokenizer.get_command_id('eos')  # end of sentence token
    cls_id = tokenizer.get_command_id('cls')  # start of sentence token
    sep_id = tokenizer.get_command_id('sep')  # seperator of two texts token

    ids = []  # ids of all the tokens
    types = [
    ]  # types of all the tokens, currently we have 0 for text a, 1 for text b,
    paddings = []  # if-is-padding of all tokens, 1 for no, 0 for yes

    # CLS
    if add_cls:
        ids.append(cls_id)
        types.append(0)  #
        paddings.append(1)

    # add text a
    len_text_a = len(text_a_ids)
    ids.extend(text_a_ids)
    types.extend([0] * len_text_a)
    paddings.extend([1] * len_text_a)

    # add text b
    if text_b_ids is not None:
        # add SEP
        if add_sep:
            ids.append(sep_id)
            types.append(0)
            paddings.append(1)
        len_text_b = len(text_b_ids)
        ids.extend(text_b_ids)
        types.extend([1] * len_text_b)
        paddings.extend([1] * len_text_b)

    eos_length = 1 if add_eos else 0

    # When size exceeds max_seq_length, cut the sequence
    if len(ids) >= max_seq_length - eos_length:
        max_seq_length_m1 = max_seq_length - 1
        ids = ids[0:max_seq_length_m1]
        types = types[0:max_seq_length_m1]
        paddings = paddings[0:max_seq_length_m1]

    # if no text_b, we also should not put 1 at the end
    end_type = 0 if text_b_ids is None else 1
    if add_eos:
        ids.append(eos_id)
        types.append(end_type)
        paddings.append(1)

    sep = len(ids)  # the position where the contents ends
    target_ids = [0] * len(ids)
    loss_masks = [0] * len(ids)
    position_ids = list(range(len(ids)))
    block_position_ids = [0] * len(ids)
    # Piece
    if add_piece or answer_ids is not None:
        sop_id = tokenizer.get_command_id('sop')
        mask_position = ids.index(
            mask_id
        ) if not args.sentinel_token else args.max_position_embeddings
        ids.append(sop_id)
        types.append(end_type)
        paddings.append(1)
        position_ids.append(mask_position)
        block_position_ids.append(1)
        if answer_ids is not None:
            len_answer = len(answer_ids)
            ids.extend(answer_ids[:-1])
            types.extend([end_type] * (len_answer - 1))
            paddings.extend([1] * (len_answer - 1))
            position_ids.extend([mask_position] * (len_answer - 1))
            if not args.no_block_position:
                block_position_ids.extend(range(2, len(answer_ids) + 1))
            else:
                block_position_ids.extend([1] * (len(answer_ids) - 1))
            target_ids.extend(answer_ids)
            loss_masks.extend([1] * len(answer_ids))
        else:
            target_ids.append(0)
            loss_masks.append(1)

    # Padding.
    padding_length = max_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([eos_id] * padding_length)
        types.extend([eos_id] * padding_length)
        paddings.extend([0] * padding_length)
        position_ids.extend([0] * padding_length)
        block_position_ids.extend([0] * padding_length)
        target_ids.extend([0] * padding_length)
        loss_masks.extend([0] * padding_length)
    if not args.masked_lm:
        position_ids = [position_ids, block_position_ids]
    return ids, types, paddings, position_ids, sep, target_ids, loss_masks


#
#
def build_decoder_input(enc_ids, answer_ids, max_seq_length,
                        max_dec_seq_length, tokenizer):
    mask_id = tokenizer.get_command_id('MASK')
    eos_id = tokenizer.get_command_id('eos')
    sop_id = tokenizer.get_command_id('sop')
    masks = []
    # TODO: it probably takes too much memory
    # for i in range(max_dec_seq_length):
    #     m = [1]*enc_len + [0]*(max_seq_length - enc_len) + [1]*(i+1) + [0]*(max_dec_seq_length-1-i)
    #     masks.append(m)
    mask_position = enc_ids.index(mask_id)
    len_answer = len(answer_ids)
    ids = [sop_id] + answer_ids[:-1]
    types = [0] * len_answer  # not used
    paddings = [1] * len_answer
    position_ids = [mask_position] * len_answer
    block_position_ids = list(range(1, len_answer + 1))
    target_ids = answer_ids
    loss_masks = [1] * len_answer
    # Padding.
    padding_length = max_dec_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([eos_id] * padding_length)
        types.extend([0] * padding_length)
        paddings.extend([0] * padding_length)
        position_ids.extend([0] * padding_length)
        block_position_ids.extend([0] * padding_length)
        target_ids.extend([0] * padding_length)
        loss_masks.extend([0] * padding_length)
    position_ids = [position_ids, block_position_ids]
    return ids, types, paddings, position_ids, masks, target_ids, loss_masks


#
#


def build_sample(ids,
                 types=None,
                 paddings=None,
                 positions=None,
                 masks=None,
                 label=None,
                 unique_id=None,
                 target=None,
                 logit_mask=None,
                 segment_ids=None,
                 loss_mask=None,
                 prompt_ids=None,
                 meta=None):
    """Convert to numpy and return a sample consumed by the batch producer."""
    #input_ids = None, position_ids = None, attention_mask = None, target_ids = None, logit_mask = None, prompt_pos = None
    ids_np = np.array(ids, dtype=np.int64)
    sample = {'input_ids': ids_np}
    if label is not None:
        labels_np = np.array(label, dtype=np.int64)
        sample['labels'] = labels_np
    if types is not None:
        types_np = np.array(types, dtype=np.int64)
        sample['types'] = types_np
    if paddings is not None:
        paddings_np = np.array(paddings, dtype=np.int64)
        sample['padding_mask'] = paddings_np
    if positions is not None:
        positions_np = np.array(positions, dtype=np.int64)
        sample['position_ids'] = positions_np
    if masks is not None:
        masks_np = np.array(masks, dtype=np.int64)
        sample['attention_mask'] = masks_np
    if target is not None:
        target_np = np.array(target, dtype=np.int64)
        sample['target_ids'] = target_np
    if logit_mask is not None:
        logit_mask_np = np.array(logit_mask, dtype=np.int64)
        sample['logit_mask'] = logit_mask_np
    if loss_mask is not None:
        loss_mask_np = np.array(loss_mask, dtype=np.int64)
        sample['loss_mask'] = loss_mask_np
    if segment_ids is not None:
        segment_ids = np.array(segment_ids, dtype=np.int64)
        sample['segment_id'] = segment_ids
    if meta is not None:
        sample['meta'] = meta
    if prompt_ids and prompt_ids is not None:
        prompt_ids = np.array(prompt_ids, dtype=np.int64)
        sample['prompt_pos'] = prompt_ids
    if unique_id is not None:
        sample['uid'] = unique_id
    return sample


def build_decoder_sample(sample, dec_ids, dec_position, dec_masks, dec_target,
                         dec_logit_mask):
    sample['dec_text'] = np.array(dec_ids)
    sample['dec_position'] = np.array(dec_position)
    sample['dec_mask'] = np.array(dec_masks)
    sample['dec_target'] = np.array(dec_target)
    sample['dec_logit_mask'] = np.array(dec_logit_mask)
    return sample


def punctuation_standardization(string: str):
    punctuation_dict = {
        "\u201c": "\"",
        "\u201d": "\"",
        "\u2019": "'",
        "\u2018": "'",
        "\u2013": "-"
    }
    for key, value in punctuation_dict.items():
        string = string.replace(key, value)
    return string


def set_deepspeed_activation_checkpointing(args):
    deepspeed.checkpointing.configure(mpu,
                                      deepspeed_config=args.deepspeed_config,
                                      num_checkpoints=args.num_layers)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # the automatic assignment of devices has been moved to arguments.py
    torch.cuda.set_device(args.device)
    print("set device")
    # Call the init process
    init_method = 'tcp://'
    # os.environ['MASTER_ADDR'] = '120.92.54.30'
    # os.environ['MASTER_PORT'] = '10501'
    args.master_ip = os.getenv('MASTER_ADDR', 'localhost')
    args.master_port = os.getenv('MASTER_PORT', '6000')
    init_method += args.master_ip + ':' + args.master_port

    torch.distributed.init_process_group(backend=args.distributed_backend,
                                         world_size=args.world_size,
                                         rank=args.rank,
                                         init_method=init_method)

    print("init distributed")
    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)
    print("set model parallel")
    # Optional DeepSpeed Activation Checkpointing Features
    if hasattr(
            args, "deepspeed"
    ) and args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(
            args)  # TODO manual model-parallel seed


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)
