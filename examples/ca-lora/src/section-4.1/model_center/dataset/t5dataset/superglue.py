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
from re import template
import torch
import json
import random

class SuperGLUE(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = []

    def make_input(self, tokenizer, template, max_encoder_length, max_decoder_length, label):
        input = tokenizer.encode(template)

        length = len(input)

        if length > max_encoder_length:
            input = input[-max_encoder_length:]

        input_tokens = torch.zeros((max_encoder_length,), dtype=torch.int32)
        input_tokens[:length] = torch.tensor(input).int()

        input_length = torch.tensor(length, dtype=torch.int32)

        output = [tokenizer.pad_token_id, tokenizer.convert_tokens_to_ids("<extra_id_0>")]
        length = len(output)
        output_tokens = torch.zeros((max_decoder_length,), dtype=torch.int32)
        output_tokens[:length] = torch.tensor(output).int()
        output_length = torch.tensor(length, dtype=torch.int32)

        target = torch.tensor(label, dtype=torch.long)

        index = torch.zeros((max_decoder_length,), dtype=torch.int32)
        index[length - 1] = 1

        self.data.append({
            "enc_input": input_tokens.cuda(),
            "enc_length": input_length.cuda(),
            "dec_input": output_tokens.cuda(),
            "dec_length": output_length.cuda(),
            "targets": target.cuda(),
            "index": index.cuda(),
        })

    def read_data(self, dataset, path, split, rank, world_size):
        if split == 'test': return
        if split == 'dev': split = 'val'
        path = f"{path}/{dataset}/{split}.jsonl"
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            for i, row in enumerate(lines):
                yield json.loads(row)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class BoolQ_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length) -> None:
        super().__init__()

        for row in self.read_data("BoolQ", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text_a = row['passage']
            text_b = row['question']

            template = f'{text_a}. {text_b}? <extra_id_0>.'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0]]


class CB_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()

        for row in self.read_data("CB", path, split, rank, world_size):
            if row["label"]=="contradiction":
                label = 0
            elif row["label"]=="entailment":
                label = 1
            else:
                label = 2
            text_a = row["premise"]
            text_b = row["hypothesis"]

            template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does sentence 1 entails sentence 2? <extra_id_0>.'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0], tokenizer.encode("Maybe")[0]]
    

class COPA_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()

        for row in self.read_data("COPA", path, split, rank, world_size):
            label = row["label"]
            text = row["premise"]
            choice_1 = row["choice1"]
            choice_2 = row["choice2"]
            question = row["question"]

            template = f'Choice 1: {choice_1} Choice 2: {choice_2} The {question} of "{text}" was choice <extra_id_0>.'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

            if split == 'train': # mirror
                label = label ^ 1
                template = f'Choice 1: {choice_2} Choice 2: {choice_1} The {question} of "{text}" was choice <extra_id_0>.'

                self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("1")[0], tokenizer.encode("2")[0]]

class COPA_MLM_Dataset(SuperGLUE):
    # MLM task but with COPA corpus
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        import random
        super().__init__()

        for row in self.read_data("COPA", path, split, rank, world_size):
            label = row["label"]
            text = row["premise"]
            choice_1 = row["choice1"]
            choice_2 = row["choice2"]
            question = row["question"]

            template = text.split(' ')
            idx = random.randint(0,len(template)-1)
            label = 0 # TODO: current version: unsupervised, check with teacher model
            # from IPython import embed; embed()
            template[idx] = '<extra_id_0>'
            template = ' '.join(template)

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        raise NotImplementedError


class MultiRC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()
        self.qids = []

        for template, label, qid in self.read_data("MultiRC", path, split, rank, world_size):
            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)
            self.qids.append(qid)

    def read_data(self, dataset, path, split, rank, world_size):
        if split == 'test': return
        if split == 'dev': split = 'val'
        path = f"{path}/{dataset}/{split}.jsonl"
        cnt = 0
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            max_id = (len(lines)) // world_size * world_size
            for i, row in enumerate(lines[:max_id]):
                row = json.loads(row)
                text = row["passage"]["text"]

                for question_json in row["passage"]["questions"]:
                    question = question_json["question"]
                    for answer_json in question_json["answers"]:
                        cnt += 1

        max_id = (cnt) // world_size * world_size
        cnt = 0
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            for i, row in enumerate(lines[:max_id]):
                row = json.loads(row)
                text = row["passage"]["text"]

                for question_json in row["passage"]["questions"]:
                    question = question_json["question"]
                    for answer_json in question_json["answers"]:
                        cnt += 1
                        if cnt > max_id: break
                        if cnt % world_size != rank: continue
                        answer = answer_json["text"]
                        label = answer_json["label"]

                        template = f'{text} Is answer "{answer}" the answer to the question "{question}"? <extra_id_0>.'

                        qid = f'{row["idx"]}-{question_json["idx"]}'

                        yield (template, label, qid)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0]]


class ReCoRD_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()

        for row in self.read_data("ReCoRD", path, split, rank, world_size):
            label = row["idx"]
            text = row["passage"]["text"]
            
            entities = []
            for entity_json in row['passage']['entities']:
                start = entity_json['start']
                end = entity_json['end']
                entity = text[start:end+1]
                entities.append(entity)

            text = text.replace("@highlight\n", "- ")  # we follow the GPT-3 paper wrt @highlight annotations

            for question_json in row["qas"]:
                question = question_json["query"]
                answers = []
                for answer_json in question_json["answers"]:
                    answer = answer_json["text"]
                    answers.append(answer)

                template = f'{text} Question: {question} Entities: {entities} Which entities can be filled in the placeholder? <extra_id_0>.'

                self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0]]


class RTE_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()

        for row in self.read_data("RTE", path, split, rank, world_size):
            label = 0 if row["label"]=="not_entailment" else 1
            text_a = row["premise"]
            text_b = row["hypothesis"]

            template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does sentence 1 entails sentence 2? <extra_id_0>.'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0]]

class RTE_MLM_Dataset(SuperGLUE):
    # MLM task but with COPA corpus
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()
        # self.masker = DataCollatorForLanguageModeling(tokenizer=tokenizer)

        for row in self.read_data("RTE", path, split, rank, world_size):
            label = 0 if row["label"]=="not_entailment" else 1
            text_a = row["premise"]
            text_b = row["hypothesis"]

            text = text_a + text_b

            template = text.split(' ')
            idx = random.randint(0,len(template)-1)

            template[idx] = '<extra_id_0>'
            template = ' '.join(template)

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)
 

    @classmethod
    def get_verbalizer(cls, tokenizer):
        raise NotImplementedError


class WiC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()

        for row in self.read_data("WiC", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text_a = row["sentence1"]
            text_b = row["sentence2"]
            word = row["word"]

            template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does the word {word} in sentence 1 express the same meaning as in sentence 2? <extra_id_0>.'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0]]


class WSC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()

        for row in self.read_data("WSC", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text = row["text"]
            
            span_1 = row["target"]["span1_text"]
            span_2 = row["target"]["span2_text"]

            template = f'{text} Does {span_2} refers to {span_1}? <extra_id_0>.'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0]]
