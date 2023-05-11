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

"""Weighted Datasets."""

import sys,random
import torch
import numpy as np

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from build_index_mappings import _build_train_valid_test_weighted_datasets
from flagai import mpu

from flagai.data.tokenizer import Tokenizer
model_dir = './'
model_name = "gpt2_new_100k"
cache_dir = model_dir + model_name
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print('tokenizer.token_end_id', tokenizer.token_end_id)

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def collate_fn(batch):
    def padding(indice, max_length, pad_idx=tokenizer.token_end_id):
        pad_indice = [
            item.tolist() + [pad_idx] * max(0, max_length - len(item.tolist())) for item in indice
        ]
        return torch.tensor(pad_indice)

    input_ids = [data["input_ids"] for data in batch]
    max_length = max([len(t) for t in input_ids])
    input_ids = padding(input_ids, max_length)[:,:seq_length]

    data = {
        "input_ids": input_ids,
        "labels": input_ids
    }
    return data

def collate_fn2(batch):
    def padding(indice, max_length, pad_idx=tokenizer.token_end_id):
        pad_indice = [
            item.tolist() + [pad_idx] * max(0, max_length - len(item.tolist())) for item in indice
        ]
        return torch.tensor(pad_indice)

    input_ids = [data for data in batch]
    max_length = max([len(t) for t in input_ids])
    input_ids = padding(input_ids, max_length)[:,:seq_length]

    data = {
        "input_ids": input_ids,
        "labels": input_ids
    }
    return data

if __name__ == '__main__':
    ## weight01, prefix01, weight02, prefix02, ...
    data_prefix = [
        2.7,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/cn_baike_text_document',
        2.91,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/cn_ebook_merge_maxlen_text_document',
        1.89,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/cn_zhihu_text_document',
        1.46,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/cn_wudao_base_text_document',
        1.01,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/cn_wudao_dedup_merged_text_document',
        0.9,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-arxiv_text_document',
        2.5,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-bookcorpus2_text_document',
        1.1,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-books3_text_document',
        1.38,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-gutenberg_pg-19_text_document',
        2.82,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-openwebtext2_text_document',
        1.01,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-pile-cc_text_document',
        0.95,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-pubmed_abstracts_text_document',
        0.95,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-pubmed_central_text_document',
        2.08,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-stackexchange_text_document',
        1.46,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-wikipedia_en_text_document',
    ]

    ## update Tokenizer add CLS & SEP tokens
    data_prefix = [
        2.7,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/cn_baike_text_document',
        2.91,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/cn_ebook_merge_maxlen_text_document',
        1.89,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/cn_zhihu_text_document',
        1.46,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/cn_wudao_base_merged_text_document',
        1.01,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/cn_wudao_dedup_merged_text_document',
        0.9,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-arxiv_text_document',
        2.5,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-bookcorpus2_text_document',
        1.1,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-books3_text_document',
        1.38,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-gutenberg_pg-19_text_document',
        2.82,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-openwebtext2_text_document',
        1.01,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-pile-cc_text_document',
        0.95,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-pubmed_abstracts_text_document',
        0.95,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-pubmed_central_text_document',
        2.08,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-stackexchange_text_document',
        1.46,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-wikipedia_en_text_document',
    ]

    '''
    '''
    ## update Tokenizer add CLS & SEP tokens
    ## add codes & update cn
    data_prefix = [
        1.0,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/cn_9_dedup_wudao_text_document',
        1.0,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/cn_9_part_merged_text_document',
        1.0,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-pile-cc_text_document',
        1.51,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-openwebtext2_text_document',

        0.6,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/code_dedup-md5-pile-github_text_document',
        0.53,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/code_code_text_document',
        0.53,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/code_newcode1_text_document',
        0.53,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/code_newcode2_text_document',
        0.38,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/code_code-cpp_text_document',
        0.38,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/code_code-java_text_document',

        1.06,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/cn_baike_text_document',
        2.43,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-wikipedia_en_text_document',

        1.0,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/cn_ebook_merge_maxlen_text_document',
        1.42,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-gutenberg_pg-19_text_document',
        1.42,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-bookcorpus2_text_document',
        1.42,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-books3_text_document',
        1.14,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-arxiv_text_document',
        1.14,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-pubmed_abstracts_text_document',

        1.13,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/cn_zhihu_text_document',
        2.08,
        '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-stackexchange_text_document',
    ]

    '''
    ## update Tokenizer add CLS & SEP tokens
    ## add codes & update cn
    ## projset
    data_prefix = [
        1.0,
        '/share/projset/LM_data/batch1_tok100k_sep/cn_9_dedup_wudao_text_document',
        1.0,
        '/share/projset/LM_data/batch1_tok100k_sep/cn_9_part_merged_text_document',
        1.0,
        '/share/projset/LM_data/batch1_tok100k_sep/en_dedup-md5-pile-pile-cc_text_document',
        1.51,
        '/share/projset/LM_data/batch1_tok100k_sep/en_dedup-md5-pile-openwebtext2_text_document',

        0.6,
        '/share/projset/LM_data/batch1_tok100k_sep/code_dedup-md5-pile-github_text_document',
        0.53,
        '/share/projset/LM_data/batch1_tok100k_sep/code_code_text_document',
        0.53,
        '/share/projset/LM_data/batch1_tok100k_sep/code_newcode1_text_document',
        0.53,
        '/share/projset/LM_data/batch1_tok100k_sep/code_newcode2_text_document',
        0.38,
        '/share/projset/LM_data/batch1_tok100k_sep/code_code-cpp_text_document',
        0.38,
        '/share/projset/LM_data/batch1_tok100k_sep/code_code-java_text_document',

        1.06,
        '/share/projset/LM_data/batch1_tok100k_sep/cn_baike_text_document',
        2.43,
        '/share/projset/LM_data/batch1_tok100k_sep/en_dedup-md5-pile-wikipedia_en_text_document',

        1.0,
        '/share/projset/LM_data/batch1_tok100k_sep/cn_ebook_merge_maxlen_text_document',
        1.42,
        '/share/projset/LM_data/batch1_tok100k_sep/en_dedup-md5-pile-gutenberg_pg-19_text_document',
        1.42,
        '/share/projset/LM_data/batch1_tok100k_sep/en_dedup-md5-pile-bookcorpus2_text_document',
        1.42,
        '/share/projset/LM_data/batch1_tok100k_sep/en_dedup-md5-pile-books3_text_document',
        1.14,
        '/share/projset/LM_data/batch1_tok100k_sep/en_dedup-md5-pile-arxiv_text_document',
        1.14,
        '/share/projset/LM_data/batch1_tok100k_sep/en_dedup-md5-pile-pubmed_abstracts_text_document',

        1.13,
        '/share/projset/LM_data/batch1_tok100k_sep/cn_zhihu_text_document',
        2.08,
        '/share/projset/LM_data/batch1_tok100k_sep/en_dedup-md5-pile-stackexchange_text_document',
    ]
    data_impl = 'mmap'
    ## splits_string len should same as train_valid_test_num_samples len
    splits_string = '9999,1'
    ## rebuilding if no npy files for train_valid_test_num_samples config
    ## 400B
    train_valid_test_num_samples = [390585937, 39063]
    seq_length = 2048
    seed = 2023
    skip_warmup = True

    '''

    data_impl = 'mmap'
    ## splits_string len should same as train_valid_test_num_samples len
    #splits_string = '9999,1'
    splits_string = '9999'
    ## rebuilding if no npy files for train_valid_test_num_samples config
    #train_valid_test_num_samples = [195312500, 19531]
    train_valid_test_num_samples = [195312500]
    seq_length = 2048
    seed = 2023
    skip_warmup = True
    ## 400 * 1000 * 1000 * 1000./ 2048 = 195312500
    train_max_num_samples = 195312500

    train_dataset, _, _ = _build_train_valid_test_weighted_datasets(
        data_prefix, data_impl, splits_string,
        train_valid_test_num_samples,
        seq_length, seed, skip_warmup, train_max_num_samples=train_max_num_samples)
    print("Total train_dataset: ", len(train_dataset))
    #print("Total valid_dataset: ", len(valid_dataset))
    #import time
    #time.sleep(10000)

    ## seed
    #set_random_seed(seed)

    def set_worker_sharing_strategy(worker_id: int) -> None:
        torch.multiprocessing.set_sharing_strategy('file_system')

    #dataset = valid_dataset
    dataset = train_dataset
    shuffle = False
    batch_size = 1
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=None,
        num_workers=4,
        drop_last=False,
        pin_memory=False,
        prefetch_factor=4,
        collate_fn=collate_fn,
        shuffle=shuffle,
        worker_init_fn=set_worker_sharing_strategy)
    
    #from megatron.data.indexed_dataset import MMapIndexedDatasetBuilder
    #builder = MMapIndexedDatasetBuilder(output_bin_file)
    from megatron.data import indexed_dataset

    num_samples = train_max_num_samples
    num_samples = 10000
    num_samples = 10000000
    output_prefix = '/share/project/ldwang/data/indexed_dataset/batch1_tok100k_sep/frozen_dataset_%d' % num_samples + 'ns'
    output_bin_file = "{}.bin".format(output_prefix)
    output_idx_file = "{}.idx".format(output_prefix)
    print(f"output_bin_file {output_bin_file}")
    builder = indexed_dataset.make_builder(output_bin_file, impl=data_impl)

    start_steps = 1000
    start_steps = 0
    end_steps = 10000000
    for iteration_, batch in enumerate(loader, start_steps):
        #if iteration_%1000==0:
        #print(f"type {type(batch['input_ids'])}")
        #sys.exit(0)
        builder.add_item(batch["input_ids"][0])
        builder.end_document()
        if iteration_ % 500 == 0:
            #print('input sample tokens=', batch["input_ids"][0].tolist(), flush=True)
            print(f"iteration_={iteration_}")
        if iteration_ == num_samples:
            break
        if False and iteration_ == end_steps:
            print(f"step={iteration_}", flush=True)
            #print('sample tokens=', batch["input_ids"][0].tolist(), flush=True)
            #print('sample decode=', tokenizer.DecodeIds(batch["input_ids"][0].tolist()), flush=True)
    builder.finalize(output_idx_file)
    print("Ended data_loader testing.")

    #from megatron.data.indexed_dataset import IndexedDataset
    from megatron.data.indexed_dataset import MMapIndexedDataset
    dataset = MMapIndexedDataset(output_prefix, skip_warmup=skip_warmup)
    print(f"Length {len(dataset)}")

    batch_size = 4
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=None,
        num_workers=4,
        drop_last=False,
        pin_memory=False,
        prefetch_factor=4,
        collate_fn=collate_fn2,
        shuffle=shuffle,
        worker_init_fn=set_worker_sharing_strategy)

    for iteration_, batch in enumerate(loader):
        if False or iteration_ == 0:
            print(f"step={iteration_}", flush=True)
            print('sample tokens=', batch["input_ids"][0].tolist(), flush=True)
