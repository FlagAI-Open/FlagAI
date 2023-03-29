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

"""GPT style dataset."""

import os
import time

import numpy as np
import torch

from megatron import mpu, print_rank_0
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.dataset_utils import get_train_valid_test_split_
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.data.gpt_dataset import _build_shuffle_idx, _build_doc_idx, _num_epochs, _num_tokens, get_indexed_dataset_

import sys
data_prefix = 'merged_text_document'
assert len(sys.argv) > 1
data_prefix = sys.argv[1]

# Indexed dataset.
dataset = get_indexed_dataset_(
    data_prefix,
    data_impl='mmap',
    skip_warmup=True)
total_num_of_documents = dataset.sizes.shape[0]

splits_string='9999,1,0'
splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
print(total_num_of_documents)
print(splits)
print(dataset[0])
print(type(dataset[0]))
print(dataset[0].shape)
last = total_num_of_documents-1
print(dataset[last])
'''

# Test Index
last = total_num_of_documents-1
result = True
try:
    dataset[last]
except:
    result = False
print(f"check\t{data_prefix}\t{result}")
'''

