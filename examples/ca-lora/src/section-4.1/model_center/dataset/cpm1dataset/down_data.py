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
    def __init__(self, path, split, rank, world_size, tokenizer, max_length) -> None:
        self.data = []

        path = f"{path}/LCQMC/{split}.tsv"
        with open(path, encoding='utf8') as fin:
            reader = list(csv.reader(fin, delimiter='\t'))[1:]
            for i, row in enumerate(reader):
                text_a, text_b, label = row
                lef_tokens = [1] + tokenizer.encode(f'"{text_a}"与"{text_b}"的关系是:')
                rig_tokens = tokenizer.encode("。")

                input_tokens, input_length, context, input_span = self.make_input(lef_tokens, rig_tokens, 1, max_length)

                index = torch.zeros((max_length,), dtype=torch.int32)
                index[len(lef_tokens) - 1] = 1

                target = torch.tensor(int(label), dtype=torch.long)

                self.data.append({
                    "input_tokens": input_tokens.cuda(),
                    "input_length": input_length.cuda(),
                    "input_context": context.cuda(),
                    "input_span": input_span.cuda(),
                    "targets": target.cuda(),
                    "index": index.cuda(),
                })

    def make_input(self, lef_tokens, rig_tokens, spans, max_length):
        input = lef_tokens + [0 for i in range(spans)] + rig_tokens
        length = len(input)

        assert length < max_length # TODO

        input_tokens = torch.zeros((max_length,), dtype=torch.int32)
        input_tokens[:length] = torch.tensor(input).int()

        input_length = torch.tensor(length, dtype=torch.int32)

        context = np.arange(max_length)
        context = (context < len(lef_tokens)) | (context >= len(lef_tokens) + spans)
        context = torch.from_numpy(context).bool()

        input_span = torch.zeros((max_length,), dtype=torch.int32)

        return input_tokens, input_length, context, input_span

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [15682, 16357] # 有关，无关 # TODO change to tokenizer.encode(xxx)