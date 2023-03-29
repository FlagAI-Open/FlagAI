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
import sys

sys.path.append("../../../")
from flagai import mpu
from flagai.data.dataset.block.lazy_loader import LazyWriter, LazyLoader, exists_lazy
from flagai.data.dataset.block.corpora import WuDaoCorpus, PromptDataset

from operator import itemgetter
from torch.utils.data import Dataset


class SplitDataset(Dataset):
    """
    Dataset wrapper to access a subset of another dataset.
    Purpose: useful to index into existing datasets, possibly
    large-scale datasets as the subindexing operation is done in an
    on-the-fly manner.
    Arguments:
        ds (Dataset or array-like): List of datasets to be subindexed
        split_inds (1D array-like): List of indices part of subset
    """

    def __init__(self, ds, split_inds, **kwargs):
        self.split_inds = list(split_inds)
        self.wrapped_data = ds
        self.is_lazy = isinstance(ds, LazyLoader) or (hasattr(ds, 'is_lazy')
                                                      and ds.is_lazy)
        self._X = None
        self._Y = None

    def __len__(self):
        return len(self.split_inds)

    def get_text_len(self, idx):
        return self.wrapped_data.get_text_len(self.split_inds[idx])

    def __getitem__(self, index):
        return self.wrapped_data[self.split_inds[index]]

    def SetTokenizer(self, tokenizer):
        self.wrapped_data.SetTokenizer(tokenizer)

    def GetTokenizer(self):
        return self.wrapped_data.GetTokenizer()

    @property
    def X(self):
        if self._X is None:
            self._X = itemgetter(*self.split_inds)(self.wrapped_data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = np.array(
                itemgetter(*self.split_inds)(self.wrapped_data.Y))
        return self._Y

    def __iter__(self):
        for idx in self.split_inds:
            yield self.wrapped_data[idx]


def split_ds(ds, split=None, shuffle=True, save_splits=None, load_splits=None):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
        shuffle (boolean): Randomly split dataset. Default: True
        save_splits: save split indices to file
        load_splits: load split indices from file
    """
    if split is None:
        split = [.8, .2, .0]
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split)
    split /= split_sum
    ds_len = len(ds)
    inds = np.arange(ds_len)
    if shuffle:
        rng = np.random.RandomState(1234)
        rng.shuffle(inds)
    # if load_splits is not None:
    #     inds = np.load(load_splits)
    #     assert len(inds) == ds_len
    # elif save_splits is not None:
    #     if torch.distributed.get_rank() == 0:
    #         np.save(save_splits, inds)
    #         print(f"Save split indices to {save_splits}")
    start_idx = 0
    residual_idx = 0
    rtn_ds = [None] * len(split)
    for i, f in enumerate(split):
        if f != 0:
            proportion = ds_len * split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            split_inds = inds[start_idx:start_idx + max(split_, 1)]
            rtn_ds[i] = SplitDataset(ds, split_inds)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds


def add_args(ds_args, tokenizer):
    if ds_args.block_lm:
        ds_args.data_set_type = "Block"
    if ds_args.sentinel_token:
        ds_args.add_sentinel_token = ds_args.max_position_embeddings
    return ds_args


def get_dataset_lazy(path,
                     tokenizer,
                     pre_tokenize,
                     num_processes,
                     no_lazy_loader=False):
    if not (exists_lazy(path, data_type='prompt')
            and exists_lazy(path, data_type='text')):
        # print(f"Creating lazy loader for dataset {name}")
        prompt_writer = LazyWriter(path,
                                   data_type='prompt',
                                   is_array=pre_tokenize)
        text_writer = LazyWriter(path, data_type='text', is_array=pre_tokenize)
        writers = {'prompt': prompt_writer, 'text': text_writer}
        reader = WuDaoCorpus(writers=writers,
                             tokenizer=tokenizer,
                             tokenize=pre_tokenize,
                             path=path)
        reader.process(num_processes)
        prompt_writer.close()
        text_writer.close()

    map_fn = (lambda x: x.tolist()) if pre_tokenize else None

    prompts = LazyLoader(path,
                         data_type='prompt',
                         map_fn=map_fn,
                         mem_map=True,
                         is_array=pre_tokenize,
                         load_memory=no_lazy_loader)
    texts = LazyLoader(path,
                       data_type='text',
                       map_fn=map_fn,
                       mem_map=True,
                       is_array=pre_tokenize,
                       load_memory=no_lazy_loader)

    text = PromptDataset(prompt_loader=prompts,
                         text_loader=texts,
                         tokenizer=tokenizer,
                         to_tokenize=not pre_tokenize)

    return text
