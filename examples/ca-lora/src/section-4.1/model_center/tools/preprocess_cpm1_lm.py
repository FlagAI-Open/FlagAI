# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
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

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np

from tokenizer import CPM1Tokenizer
import indexed_dataset

random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = CPM1Tokenizer(os.path.join(self.args.tokenizer_path, 'vocab.txt'))

    def encode(self, line):
        # end with <eod>
        if len(line) > 5000000:
            return None, None, 0

        data = line.strip()
        data = data.replace("<n>", "\n")

        doc_ids = Encoder.tokenizer.encode(data)
        if len(doc_ids) < 32:
            return None, None, 0

        doc_ids.append(Encoder.tokenizer.eod_id)
        doc_ids = [1] + doc_ids
        doc_ids = [j for j in doc_ids if j != Encoder.tokenizer.unk_id]

        contexts = []
        labels = []
        i = 0

        while i < len(doc_ids):
            piece = doc_ids[i:i+512]
            if len(piece) < 32:
                break
            i += 512

            context = piece
            label = piece
            assert len(label) == len(context)
            assert len(label) <= 512
            contexts.append(context)
            labels.append(label)

        return contexts, labels, len(line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', default="/mnt/sfs_turbo/new_data/cpm1_data/cpm1_", type=str, help='Path to input TXT')
    
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', default="/mnt/sfs_turbo/ModelCenter/vocab/new_cn/", type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="/mnt/sfs_turbo/ModelCenter/new_data/", type=str)
    group.add_argument('--output_prefix', default="cpm1_lm", type=str, help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap', choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--uid', type=int, default=0,
                       help='Number of worker processes to launch')
    group.add_argument('--workers', type=int, default=64,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.make_vocab_size_divisible_by = 128

    return args

def main():
    args = get_args()
    startup_start = time.time()

    uid = args.uid
    print("Opening", args.input+str(uid)+".txt")
    fin = open(args.input+str(uid)+".txt", 'r', encoding='utf-8')

    encoder = Encoder(args)
    # tokenizer = CPM1Tokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'), space_token = "▂", line_token = "▃")
    # pool = Pool(args.workers, initializer=encoder.initializer)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)

    level = "document"

    # print(f"Vocab size: {encoder.initializer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")

    context_bin_file = os.path.join(args.output_path, "{}_{}_context_{}.bin".format(args.output_prefix, level, uid))
    context_idx_file = os.path.join(args.output_path,  "{}_{}_context_{}.idx".format(args.output_prefix, level, uid))
    # target_bin_file = os.path.join(args.output_path,  "{}_{}_target_{}.bin".format(args.output_prefix, level, uid))
    # target_idx_file = os.path.join(args.output_path,  "{}_{}_target_{}.idx".format(args.output_prefix, level, uid))
    
    builder_context = indexed_dataset.make_builder(context_bin_file, impl=args.dataset_impl, dtype=np.int32)
    # builder_target = indexed_dataset.make_builder(target_bin_file, impl=args.dataset_impl, dtype=np.int32)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    # sentinel_idx = tokenizer.vocab_size # start from the last token of the tokenizer
    # print("tokenizer vocab size:", encoder.initializer.vocab_size)
    for i, (pair_ids, label_ids, bytes_processed) in enumerate(encoded_docs, start=1):
        if pair_ids is None or label_ids is None:
            continue
        total_bytes_processed += bytes_processed

        for pids, lids in zip(pair_ids, label_ids):
            builder_context.add_item(torch.IntTensor(pids))
            # builder_target.add_item(torch.IntTensor(lids))
        
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    builder_context.finalize(context_idx_file)
    # builder_target.finalize(target_idx_file)

    pool.close()

if __name__ == '__main__':
    main()
