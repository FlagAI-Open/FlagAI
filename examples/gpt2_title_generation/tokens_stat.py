# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset

model_dir = "./state_dict/"
maxlen = 1024

from flagai.data.tokenizer import Tokenizer
model_name = "GPT2-xlarge-en"
cache_dir = model_dir + model_name
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)

def read_file():
    part_file = '/share/project/ldwang/data/pile/train/00.txt'
    part_file = '00.txt'
    total = 0
    count = 0
    if True:
        filename = part_file
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                count += 1
                src = line.strip('\n').lower()
                data = tokenizer.encode_plus(src, src, max_length=maxlen)
                size = len(data['input_ids'])
                print('size', size)
                if size <= maxlen:
                    total += size
                else:
                    total += maxlen
    print(total, count, total*1.0/count)

read_file()

