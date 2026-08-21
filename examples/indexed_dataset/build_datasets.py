# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset
from flagai.data.dataset.indexed_dataset.build_datasets import _build_train_valid_test_datasets

data_prefix = 'data/demo_text_document'
data_impl = 'mmap'
splits_string = '90,10'
train_valid_test_num_samples = [90, 10]
seq_length = 1024
seed = 2023
skip_warmup = True

train_dataset, val_dataset, _ = _build_train_valid_test_datasets(
    data_prefix, data_impl, splits_string,
    train_valid_test_num_samples,
    seq_length, seed, skip_warmup)
