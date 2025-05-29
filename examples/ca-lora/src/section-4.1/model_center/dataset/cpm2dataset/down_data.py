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
import torch
import csv
import numpy as np

class LCQMC_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length) -> None:
        self.data = []

        path = f"{path}/LCQMC/{split}.tsv"
        with open(path, encoding='utf8') as fin:
            reader = list(csv.reader(fin, delimiter='\t'))[1:]
            for i, row in enumerate(reader):
                text_a, text_b, label = row
                enc_input = tokenizer.encode(f'“{text_a}”与“{text_b}”是否有关？')

                enc_tokens, enc_length, dec_tokens, dec_length, index = self.make_input(tokenizer, enc_input, max_encoder_length, max_decoder_length)

                target = torch.tensor(int(label), dtype=torch.long)

                self.data.append({
                    "enc_input": enc_tokens.cuda(),
                    "enc_length": enc_length.cuda(),
                    "dec_input": dec_tokens.cuda(),
                    "dec_length": dec_length.cuda(),
                    "targets": target.cuda(),
                    "index": index.cuda(),
                })

    def make_input(self, tokenizer, input, max_encoder_length, max_decoder_length):
        input = input + [tokenizer.get_sentinel_id(0)]
        length = len(input)

        assert length < max_encoder_length # TODO

        input_tokens = torch.zeros((max_encoder_length,), dtype=torch.int32)
        input_tokens[:length] = torch.tensor(input).int()

        input_length = torch.tensor(length, dtype=torch.int32)

        output = [tokenizer.get_sentinel_id(0)]
        length = len(output)
        output_tokens = torch.zeros((max_decoder_length,), dtype=torch.int32)
        output_tokens[:length] = torch.tensor(output).int()
        output_length = torch.tensor(length, dtype=torch.int32)

        index = torch.zeros((max_decoder_length,), dtype=torch.int32)
        index[length - 1] = 1

        return input_tokens, input_length, output_tokens, output_length, index

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [1744, 24] # 有关，无关 # TODO change to tokenizer.encode(xxx)