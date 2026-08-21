# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
from abc import abstractclassmethod
import torch
import json

class SuperGLUE(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = []

    def make_input(self, tokenizer, template, max_encoder_length, label):
        input = tokenizer(
            *template, 
            max_length=max_encoder_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        labels = torch.tensor(label, dtype=torch.long)

        self.data.append({
            "input_ids": input['input_ids'][0].cuda(),
            "attention_mask": input['attention_mask'][0].cuda(),
            "token_type_ids": input['token_type_ids'][0].cuda(),
            "labels": labels.cuda(),
        })

    def make_double_input(self, tokenizer, template0, template1, max_encoder_length, label): # for COPA dataset
        input0 = tokenizer(
            *template0, 
            max_length=max_encoder_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        input1 = tokenizer(
            *template1, 
            max_length=max_encoder_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        labels = torch.tensor(label, dtype=torch.long)

        self.data.append({
            "input_ids0": input0['input_ids'][0].cuda(),
            "attention_mask0": input0['attention_mask'][0].cuda(),
            "token_type_ids0": input0['token_type_ids'][0].cuda(),
            "input_ids1": input1['input_ids'][0].cuda(),
            "attention_mask1": input1['attention_mask'][0].cuda(),
            "token_type_ids1": input1['token_type_ids'][0].cuda(),
            "labels": labels.cuda(),
        })

    def read_data(self, dataset, path, split, rank, world_size):
        if split == 'test': return
        if split == 'dev': split = 'val'
        path = f"{path}/{dataset}/{split}.jsonl"
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            for i, row in enumerate(lines):
                yield json.loads(row)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class BoolQ_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length) -> None:
        super().__init__()

        from tqdm import tqdm
        for row in self.read_data("BoolQ", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text_a = row['passage']
            text_b = row['question']

            # template = (f'{text_a}', f'{text_b}')
            template = (f'{text_a}. {text_b}',)

            self.make_input(tokenizer, template, max_encoder_length, label)


class CB_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()

        count = [0, 0, 0]
        for row in self.read_data("CB", path, split, rank, world_size):
            if row["label"]=="contradiction":
                label = 0
            elif row["label"]=="entailment":
                label = 1
            else:
                label = 2
            count[label] += 1
        mx = max(count)
        for i in range(3):
            if(count[i] == 0):
                continue
            count[i] = int(mx / count[i])
            print('count[', i, '] = ', count[i])

        new_count = [0, 0, 0]

        for row in self.read_data("CB", path, split, rank, world_size):
            if row["label"]=="contradiction":
                label = 0
            elif row["label"]=="entailment":
                label = 1
            else:
                label = 2
            text_a = row["premise"]
            text_b = row["hypothesis"]

            template = (f'{text_a}', f'{text_b}')

            if split == 'train':
                for i in range(count[label]):
                    self.make_input(tokenizer, template, max_encoder_length, label)
                    new_count[label] += 1
            else:
                self.make_input(tokenizer, template, max_encoder_length, label)

        print('count:', count, ' | new_count :', new_count)
    

class COPA_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()

        for row in self.read_data("COPA", path, split, rank, world_size):
            label = row["label"]
            text = row["premise"]
            choice_1 = row["choice1"]
            choice_2 = row["choice2"]
            question = row["question"]

            template0 = (f'{choice_1}', f'The {question} of "{text}"')
            template1 = (f'{choice_2}', f'The {question} of "{text}"')
            self.make_double_input(tokenizer, template0, template1, max_encoder_length, label)


class RTE_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()

        for row in self.read_data("RTE", path, split, rank, world_size):
            label = 0 if row["label"]=="not_entailment" else 1
            text_a = row["premise"]
            text_b = row["hypothesis"]

            #template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does sentence 1 entails sentence 2?'
            template = (f'{text_a}', f'{text_b}')

            self.make_input(tokenizer, template, max_encoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode(" No")[0], tokenizer.encode(" Yes")[0]]


class WiC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()

        for row in self.read_data("WiC", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text_a = row["sentence1"]
            text_b = row["sentence2"]
            word = row["word"]

            #template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does the word {word} in sentence 1 express the same meaning as in sentence 2?'
            template = (f'{text_a}', f'{text_b}')

            self.make_input(tokenizer, template, max_encoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode(" No")[0], tokenizer.encode(" Yes")[0]]


class WSC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()

        for row in self.read_data("WSC", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text = row["text"]
            
            span_1 = row["target"]["span1_text"]
            span_2 = row["target"]["span2_text"]

            template = (f'{text} Does {span_2} refers to {span_1}?',)

            self.make_input(tokenizer, template, max_encoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode(" No")[0], tokenizer.encode(" Yes")[0]]
