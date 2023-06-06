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
    batch1_tok100k = '/share/project/ldwang/data/indexed_dataset/batch1_tok100k'
    batch2_tok100k = '/share/project/ldwang/data/indexed_dataset/batch2_tok100k'
    data_prefix = [
        2.242990654,
        os.path.join(batch1_tok100k, 'en_dedup-md5-pile-openwebtext2_text_document'),
        5.046728972,
        os.path.join(batch1_tok100k, 'en_dedup-md5-pile-pile-cc_text_document'),
        13.64485981,
        os.path.join(batch1_tok100k, 'wudao-9_text_document'),
        2.336448598,
        os.path.join(batch1_tok100k, 'code_dedup-md5-pile-github_text_document'),
        1.869158879,
        os.path.join(batch1_tok100k, 'codegeex_text_document'),
        1.588785047,
        os.path.join(batch1_tok100k, 'en_dedup-md5-pile-wikipedia_en_text_document'),
        2.336448598,
        os.path.join(batch1_tok100k, 'cn_baike_text_document'),
        4.205607477,
        os.path.join(batch1_tok100k, 'pile-books_text_document'),
        0.186915888,
        os.path.join(batch1_tok100k, 'cn_ebook_merge_maxlen_text_document'),
        2.429906542,
        os.path.join(batch1_tok100k, 'pile-papers_text_document'),
        1.869158879,
        os.path.join(batch1_tok100k, 'en_dedup-md5-pile-stackexchange_text_document'),
        0.747663551,
        os.path.join(batch1_tok100k, 'cn_zhihu_text_document'),

        31.77570093,
        os.path.join(batch2_tok100k, 'ccnews_text_document'),
        12.42990654,
        os.path.join(batch2_tok100k, 'c4_text_document'),
        11.58878505,
        os.path.join(batch2_tok100k, 'wudao-3-8_text_document'),
        1.869158879,
        os.path.join(batch2_tok100k, 'hf-wiki_text_document'),
        0.654205607,
        os.path.join(batch2_tok100k, 'sjt_text_document'),
        1.214953271,
        os.path.join(batch2_tok100k, 'col_text_document'),
        1.121495327,
        os.path.join(batch2_tok100k, 'byg-cn_text_document'),
        0.093457944,
        os.path.join(batch2_tok100k, 'qa_text_document'),
        0.747663551,
        os.path.join(batch2_tok100k, 'wenge-zhihu-high_text_document'),
    ]

    data_impl = 'mmap'
    ## splits_string len should same as train_valid_test_num_samples len
    splits_string = '9999,1'
    ## 2. specify total samples needed
    ## 400B = 400 * 1000 * 1000 * 1000./ 2048 = 195312500
    ## 1000B = 1000 * 1000 * 1000 * 1000./ 2048 = 488281250
    train_max_num_samples = 195312500
    train_max_num_samples = 488281250
    ## 1070B
    train_max_num_samples = 522460937
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

