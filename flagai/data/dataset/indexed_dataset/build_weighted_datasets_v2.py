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

import sys, os
import argparse
import torch

from flagai.data.dataset.indexed_dataset.build_datasets import _build_train_valid_test_weighted_datasets
from flagai.data.tokenizer import Tokenizer

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='build weighted datasets')
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--model-name', type=str, required=True, default=None,
                       help='What model to use.')
    group.add_argument('--model-dir', type=str, required=True, default=None,
                       help='What model dir to use.')
    args = parser.parse_args()
    return args

args = get_args()
cache_dir = os.path.join(args.model_dir, args.model_name)
tokenizer = Tokenizer.from_pretrained(args.model_name,
                                      cache_dir=cache_dir)
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

if __name__ == '__main__':
    ## 1: weight01, prefix01, weight02, prefix02, ...
    data_prefix = [
        1.296091,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_9_dedup_wudao_text_document',
        144.325674,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_9_part_merged_text_document',
        53.498074,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-pile-cc_text_document',
        23.575721,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-openwebtext2_text_document',
    
        14.718128,
        '/data/indexed_dataset/batch1_tok100k_sep/code_dedup-md5-pile-github_text_document',
        8.878174,
        '/data/indexed_dataset/batch1_tok100k_sep/code_code_text_document',
        3.439587,
        '/data/indexed_dataset/batch1_tok100k_sep/code_newcode1_text_document',
        2.533595,
        '/data/indexed_dataset/batch1_tok100k_sep/code_newcode2_text_document',
        9.410141,
        '/data/indexed_dataset/batch1_tok100k_sep/code_code-cpp_text_document',
        5.965614,
        '/data/indexed_dataset/batch1_tok100k_sep/code_code-java_text_document',
    
        22.442690,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_baike_text_document',
        10.276255,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-wikipedia_en_text_document',
    
        6.821143,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_ebook_merge_maxlen_text_document',
        4.057581,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-gutenberg_pg-19_text_document',
        2.266030,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-bookcorpus2_text_document',
        37.479110,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-books3_text_document',
        20.044762,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-arxiv_text_document',
        4.826957,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-pubmed_abstracts_text_document',
    
        7.514409,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_zhihu_text_document',
        19.639909,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-stackexchange_text_document',
    ]
    

    data_impl = 'mmap'
    ## splits_string len should same as train_valid_test_num_samples len
    splits_string = '9999,1'
    ## 2. specify total samples needed
    ## 400B = 400 * 1000 * 1000 * 1000./ 2048 = 195312500
    ## 1000B = 1000 * 1000 * 1000 * 1000./ 2048 = 488281250
    train_max_num_samples = 195312500
    train_valid_test_num_samples = [train_max_num_samples, int(train_max_num_samples*0.00001)]
    seq_length = 2048
    seed = 2023
    skip_warmup = True

    train_dataset, valid_dataset, _ = _build_train_valid_test_weighted_datasets(
        data_prefix, data_impl, splits_string,
        train_valid_test_num_samples,
        seq_length, seed, skip_warmup, train_max_num_samples=train_max_num_samples)
    print("Total train_dataset: ", len(train_dataset))
    print("Total valid_dataset: ", len(valid_dataset))

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        sampler=None,
        num_workers=16,
        drop_last=False,
        pin_memory=False,
        prefetch_factor=4,
        collate_fn=collate_fn)
    
    ## a little testing
    for iteration_, batch in enumerate(loader, 0):
        if iteration_%50000==0:
            print(f"step={iteration_}", flush=True)
    print("Ended data_loader testing.")

