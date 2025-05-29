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

"""Pretrain BERT"""

from functools import partial

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import BertModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building BERT model ...')

    args = get_args()
    args.custom_token_counting = True
    num_tokentypes = 2 if args.bert_binary_head else 0
    post_process = False
    model = BertModel(
        num_tokentypes=num_tokentypes,
        add_binary_head=args.bert_binary_head,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process)

    return model


_x = 0

def forward_step_test(data_iterator, model):
    args = get_args()
    timers = get_timers()

    tokens = torch.randint(30522, (args.micro_batch_size, args.seq_length), dtype = torch.long, device = "cuda")
    att_mask = torch.ones((args.micro_batch_size, args.seq_length), dtype = torch.long, device = "cuda")
    labels = torch.randint(args.hidden_size, (args.micro_batch_size, ), dtype = torch.long, device = "cuda")

    timers("_forward").start()
    output_tensor = model(tokens, att_mask)
    timers("_forward").stop()

    def loss_f(labels, logits):
        ce = torch.nn.CrossEntropyLoss(ignore_index = -100)
        loss = ce(logits, labels)
        averaged_losses = average_losses_across_data_parallel_group([loss])
        return loss, {'lm loss': averaged_losses[0]}

    return output_tensor[0][0, :, :], partial(loss_f, labels)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        binary_head=args.bert_binary_head)
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step_test,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
