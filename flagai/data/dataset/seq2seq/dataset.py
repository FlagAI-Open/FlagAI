# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import json
import random
from re import T
import torch.utils.data
import numpy as np
from ..data_utils import InputExample
import zipfile

print_rank_0 = print


def punctuation_standardization(string: str):
    punctuation_dict = {
        "\u201c": "\"",
        "\u201d": "\"",
        "\u2019": "'",
        "\u2018": "'",
        "\u2013": "-"
    }
    for key, value in punctuation_dict.items():
        string = string.replace(key, value)
    return string


def gigaword_detokenize(string, is_target=False):
    _tok_dict = {
        "(": "-lrb-",
        ")": "-rrb-",
        "[": "-lsb-",
        "]": "-rsb-",
        "{": "-lcb-",
        "}": "-rcb-",
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;'
    }
    string = string.replace('UNK', '[UNK]')
    string = string.replace('<unk>', '[UNK]')
    for key, value in _tok_dict.items():
        string = string.replace(value, key)
    # string = string.replace("''", "\"")
    # string = string.replace("``", "\"")
    # string = string.replace("`", "'")
    # string = string.replace(" n't", "n't")
    # string = string.replace(" 's", "'s")
    # string = string.replace(" 'd", "'d")
    # string = string.replace(" 'll", "'ll")
    return string


def cnndm_detokenize(string, is_target=False):
    _tok_dict = {
        "(": "-LRB-",
        ")": "-RRB-",
        "[": "-LSB-",
        "]": "-RSB-",
        "{": "-LCB-",
        "}": "-RCB-"
    }
    if not is_target:
        string = string.replace("<S_SEP>", "")
    else:
        string = string.replace("<S_SEP>", "[SEP]")
    for key, value in _tok_dict.items():
        string = string.replace(value, key)
    string = string.replace("''", "\"")
    string = string.replace("``", "\"")
    string = string.replace("`", "'")
    string = string.replace(" n't", "n't")
    string = string.replace(" 's", "'s")
    string = string.replace(" 'd", "'d")
    string = string.replace(" 'll", "'ll")
    return string


def blanklm_detokenize(string, is_target=False):
    string = string.replace("_UNK", "[UNK]")
    string = string.replace("<blank>", "[MASK]")
    return string


class SummmaryProcessor:

    def __init__(self, task_name, data_dir, tokenizer):
        self.task_name = task_name
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def create_examples(self, split):
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "val"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        print_rank_0(
            f"Creating {self.task_name}-{split} dataset from {self.data_dir}")
        if self.task_name == "gigaword":
            detokenizer = gigaword_detokenize
        elif self.task_name in ["cnn_dm", "lang8_hsk"]:
            detokenizer = cnndm_detokenize
        else:
            detokenizer = None
        source_texts, target_texts = [], []
        with open(os.path.join(self.data_dir + "/" + self.task_name,
                               f"{filename}.source"),
                  encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = detokenizer(line) if detokenizer else line
                source_texts.append(line)
        with open(os.path.join(self.data_dir + "/" + self.task_name,
                               f"{filename}.target"),
                  encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = detokenizer(line,
                                   is_target=True) if detokenizer else line
                target_texts.append(line)
        assert len(source_texts) == len(target_texts)
        example_list = []
        for idx, (source_text,
                  target_text) in enumerate(zip(source_texts, target_texts)):
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % (split, idx)
            meta = {
                "ref":
                self.tokenizer.DecodeIds(
                    self.tokenizer.EncodeAsIds(target_text))
            }
            example = InputExample(guid=guid,
                                   text_a=source_text,
                                   text_b=target_text,
                                   meta=meta)
            if idx < 1:
                print_rank_0(
                    (source_text.encode('utf-8'), target_text.encode('utf-8'),
                     meta["ref"].encode('utf-8')))
            example_list.append(example)
        
        return example_list


class WSCProcessor:

    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def create_examples(self, split):
        if split == "train":
            filename = "train.jsonl"
        elif split == "dev":
            filename = "val.jsonl"
        elif split == "test":
            filename = "test.jsonl"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating CMRC-{split} dataset from {self.data_dir}")
        example_list = []
        idx = 0
        with open(os.path.join(self.data_dir, filename),
                  encoding='utf-8') as file:
            for line in file.readlines():
                content = json.loads(line)
                context = content['text']
                guid = "%s-%s" % (split, idx)
                target = content['target']
                meta = {
                    "answer":
                    target['span1_text'],
                    "question":
                    target['span2_text'],
                    "ref":
                    self.tokenizer.DecodeIds(
                        self.tokenizer.EncodeAsIds(target['span1_text']))
                }
                example = InputExample(guid=guid, text_a=context, meta=meta)
                example_list.append(example)
                idx += 1

        print_rank_0(f"Creating {len(example_list)} examples for {split}")
        return example_list


class CMRCProcessor:

    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def create_examples(self, split):
        if split == "train":
            filename = "train.json"
        elif split == "dev":
            filename = "dev.json"
        elif split == "test":
            filename = "test.json"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating CMRC-{split} dataset from {self.data_dir}")
        example_list = []
        idx = 0
        with open(os.path.join(self.data_dir, filename),
                  encoding='utf-8') as file:
            dataset = json.load(file)
            for article in dataset['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa["question"]
                        answers = {answer['text']
                                   for answer in qa["answers"]
                                   } if split != 'test' else {"FAKE_ANSWER"}
                        for answer in answers:
                            guid = "%s-%s" % (split, idx)
                            meta = {
                                "answer":
                                answer,
                                "question":
                                question,
                                "ref":
                                self.tokenizer.DecodeIds(
                                    self.tokenizer.EncodeAsIds(answer))
                            }
                            example = InputExample(guid=guid,
                                                   text_a=context,
                                                   meta=meta)
                            # if idx < 10:
                            #     print_rank_0((context.encode('utf-8'),
                            #                   answer.encode('utf-8'),
                            #                   meta["ref"].encode('utf-8')))
                            example_list.append(example)
                            idx += 1
        print_rank_0(f"Creating {len(example_list)} examples for {split}")
        return example_list


class SQuADProcessor:

    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def create_examples(self, split):
        if split == "train":
            filename = "train.json"
        elif split == "dev":
            filename = "dev.json"
        elif split == "test":
            filename = "test.json"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating SQuAD-{split} dataset from {self.data_dir}")
        example_list = []
        idx = 0
        with open(os.path.join(self.data_dir, filename),
                  encoding='utf-8') as file:
            dataset = json.load(file)
            for paragraphs in dataset:
                for paragraph in paragraphs['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa["question"]
                        answers = {answer["text"] for answer in qa["answers"]}
                        answer_starts = {
                            answer["text"]: answer["answer_start"]
                            for answer in qa["answers"]
                        }
                        for answer in answers:
                            guid = "%s-%s" % (split, idx)
                            meta = {
                                "answer_start":
                                answer_starts[answer],
                                "answer":
                                answer,
                                "question":
                                question,
                                "ref":
                                self.tokenizer.DecodeIds(
                                    self.tokenizer.EncodeAsIds(question))
                            }
                            example = InputExample(guid=guid,
                                                   text_a=context,
                                                   meta=meta)
                            if idx < 10:
                                print_rank_0((context.encode('utf-8'),
                                              answer.encode('utf-8'),
                                              meta["ref"].encode('utf-8')))
                            example_list.append(example)
                            idx += 1
        print_rank_0(f"Creating {len(example_list)} examples for {split}")
        return example_list


class XSumProcessor:

    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def create_examples(self, split):
        if split == "train":
            key = "train"
        elif split == "dev":
            key = "validation"
        elif split == "test":
            key = "test"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating XSUM-{split} dataset from {self.data_dir}")
        with open(
                os.path.join(
                    self.data_dir,
                    "XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json")) as file:
            id_list = json.load(file)
        id_list = id_list[key]
        source_texts, target_texts = [], []
        for i, idx in enumerate(id_list):
            with open(os.path.join(self.data_dir, f"{idx}.summary")) as file:
                key, sentences = None, []
                source_text, target_text = None, None
                for line in file:
                    line = line.strip()
                    if line.startswith("[SN]"):
                        if key is not None:
                            if key == "RESTBODY":
                                source_text = " ".join(sentences)
                            elif key == "FIRST-SENTENCE":
                                target_text = " ".join(sentences)
                        key = line[4:-4]
                        sentences = []
                    elif line:
                        sentences.append(line)
                if key is not None:
                    if key == "RESTBODY":
                        source_text = " ".join(sentences)
                    elif key == "FIRST-SENTENCE":
                        target_text = " ".join(sentences)
                source_texts.append(source_text)
                target_texts.append(target_text)
                if (i + 1) % 1000 == 0:
                    print_rank_0(f"Complete {i + 1} examples")
        assert len(source_texts) == len(target_texts)
        example_list = []
        for idx, (source_text,
                  target_text) in enumerate(zip(source_texts, target_texts)):
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % (split, idx)
            meta = {
                "ref":
                self.tokenizer.DecodeIds(
                    self.tokenizer.EncodeAsIds(target_text))
            }
            example = InputExample(guid=guid,
                                   text_a=source_text,
                                   text_b=target_text,
                                   meta=meta)
            if idx < 10:
                print_rank_0(
                    (source_text.encode('utf-8'), target_text.encode('utf-8'),
                     meta["ref"].encode('utf-8')))
            example_list.append(example)
        return example_list


from flagai.data.dataset.superglue.control import SuperGlueProcessor


class Seq2SeqDataset(torch.utils.data.Dataset):

    def __init__(self,
                 task_name='lang8_hsk',
                 data_dir='./data',
                 dataset_type='train',
                 tokenizer=None,
                 no_block_position=False,
                 task_mask=False,
                 src_seq_length=608,
                 tgt_seq_length=160):
        self.task_name = task_name
        self.data_dir = data_dir 
        self.max_src_length, self.max_tgt_length = src_seq_length, tgt_seq_length
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.dataset_name = dataset_type
        if not os.path.exists(data_dir + '/' + task_name):
            SuperGlueProcessor()._download_data(data_dir, task_name)

        if self.task_name in [
                "gigaword", "cnn_dm", "cnn_dm_original", "lang8_hsk"
        ]:
            self.processor = SummmaryProcessor(self.task_name, self.data_dir,
                                               tokenizer)
        elif self.task_name in ["xsum"]:
            self.processor = XSumProcessor(self.data_dir+task_name, tokenizer)
        elif self.task_name in ["squad_generation"]:
            self.processor = SQuADProcessor(self.data_dir+task_name, tokenizer)
        elif self.task_name in ['cmrc']:
            self.processor = CMRCProcessor(self.data_dir+task_name, tokenizer)
        elif self.task_name in ['wsc']:
            self.processor = WSCProcessor(self.data_dir+task_name, tokenizer)
        else:
            raise NotImplementedError(self.task_name)
        example_list = self.processor.create_examples(dataset_type)
        self.example_list = example_list
        self.examples = {example.guid: example for example in example_list}
        self.task_mask = task_mask
        self.no_block_position = no_block_position
        print_rank_0(f"Return {len(self.examples)} {dataset_type} examples")

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        return example

class ExtractionDataset(torch.utils.data.Dataset):

    def __init__(self, args, split, tokenizer):
        self.args = args
        task = args.task.lower()
        data_dir = args.data_dir + task + '/'
        self.max_src_length, self.max_tgt_length = args.src_seq_length, args.tgt_seq_length
        self.split = split
        self.tokenizer = tokenizer
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "valid"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating {task}-{split} dataset from {data_dir}")
        self.dataset_name = split
        source_texts, target_texts = [], []
        with open(os.path.join(data_dir, f"{filename}.source"),
                  encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                source_texts.append(line)
        with open(os.path.join(data_dir, f"{filename}.target"),
                  encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                target_texts.append(line)
        self.examples, self.example_list = {}, []
        for idx, (source_text,
                  target_text) in enumerate(zip(source_texts, target_texts)):
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % (split, idx)
            meta = {"ref": target_text}
            example = InputExample(guid=guid,
                                   text_a=source_text,
                                   text_b=target_text,
                                   meta=meta)
            self.examples[guid] = example
            self.example_list.append(example)
        print_rank_0(f"Return {len(self.examples)} {split} examples")

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        source_text, target_text = example.text_a, example.text_b
        mask_token = 'MASK'
        mask_id = self.tokenizer.get_command_id(mask_token)
        sop_id = self.tokenizer.get_command_id('sop')
        eop_id = self.tokenizer.get_command_id('eop')
        pad_id = self.tokenizer.get_command_id('pad')

        def pad_to(text, max_len, pad_id):
            if len(text) > max_len:
                text = text[:max_len]
            else:
                text = text + [pad_id] * (max_len - len(text))
            return text

        source_tokens = self.tokenizer.EncodeAsIds(source_text)
        masked_tgt = target_text.split("|")
        source_tokens = pad_to(source_tokens, self.max_src_length, pad_id)
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        if self.split == 'train':
            mask_positions = [
                i for i, x in enumerate(source_tokens) if x == mask_id
            ]
            assert len(mask_positions) <= len(masked_tgt)
            tokens = source_tokens
            target_ids = [0] * len(source_tokens)
            loss_mask = [0] * len(source_tokens)
            for i, mask_pos in enumerate(mask_positions):
                tgt_text = masked_tgt[i]
                tgt_tokens = self.tokenizer.EncodeAsIds(" " + tgt_text)
                tokens += [sop_id] + tgt_tokens
                target_ids += tgt_tokens + [eop_id]
                loss_mask += [1] * (len(tgt_tokens) + 1)
                position_ids += [mask_pos] * (len(tgt_tokens) + 1)
                block_position_ids += [
                    i + 1 for i in range(len(tgt_tokens) + 1)
                ]
            tokens = pad_to(tokens, self.max_src_length + self.max_tgt_length,
                            pad_id)
            target_ids = pad_to(target_ids,
                                self.max_src_length + self.max_tgt_length,
                                pad_id)
            loss_mask = pad_to(loss_mask,
                               self.max_src_length + self.max_tgt_length, 0)
            position_ids = pad_to(position_ids,
                                  self.max_src_length + self.max_tgt_length, 0)
            block_position_ids = pad_to(
                block_position_ids, self.max_src_length + self.max_tgt_length,
                0)
            position_ids = [position_ids, block_position_ids]
            sample = {
                'input_ids': np.array(tokens, dtype=np.int64),
                'target': np.array(target_ids, dtype=np.int64),
                'attention_mask': np.array(sep, dtype=np.int64),
                'loss_mask': np.array(loss_mask, dtype=np.int64),
                "position_id": np.array(position_ids, dtype=np.int64),
                "uid": example.guid
            }
        else:
            tokens = source_tokens + [sop_id]
            mask_pos = source_tokens.index(mask_id)
            position_ids = position_ids + [mask_pos]
            block_position_ids = block_position_ids + [1]
            position_ids = [position_ids, block_position_ids]
            sample = {
                'input_ids': np.array(tokens, dtype=np.int64),
                'attention_mask': np.array(sep, dtype=np.int64),
                "position_id": np.array(position_ids, dtype=np.int64),
                "uid": example.guid
            }
        return sample


class BlankLMDataset(torch.utils.data.Dataset):

    def __init__(self, args, split, tokenizer):
        self.args = args
        task, data_dir = args.task.lower(), args.data_dir
        self.max_src_length, self.max_tgt_length = args.src_seq_length, args.tgt_seq_length
        self.split = split
        assert args.tokenizer_type == "GLMBertWordPieceTokenizer"
        self.tokenizer = tokenizer
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "valid"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating {task}-{split} dataset from {data_dir}")
        self.dataset_name = split
        detokenizer = blanklm_detokenize
        source_texts, target_texts = [], []
        with open(os.path.join(data_dir, f"{filename}.txt"),
                  encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = detokenizer(line) if detokenizer else line
                target_texts.append(line)
        if split == 'test':
            with open(os.path.join(
                    data_dir,
                    f"blank/test.maskratio{args.blank_maskratio:.1f}.blank"),
                      encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    line = detokenizer(line) if detokenizer else line
                    source_texts.append(line)
        else:
            source_texts = target_texts
        self.examples, self.example_list = {}, []
        for idx, (source_text,
                  target_text) in enumerate(zip(source_texts, target_texts)):
            # if idx > 10000:
            #     break
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % (split, idx)
            meta = {"ref": target_text}
            example = InputExample(guid=guid,
                                   text_a=source_text,
                                   text_b=target_text,
                                   meta=meta)
            self.examples[guid] = example
            self.example_list.append(example)
        print_rank_0(f"Return {len(self.examples)} {split} examples")
        self.random = random.Random(args.seed)

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        source_text = example.text_a
        mask_token = 'gMASK' if self.args.task_mask else 'MASK'
        mask_id = self.tokenizer.get_command_id(mask_token)
        sop_id = self.tokenizer.get_command_id('sop')
        eop_id = self.tokenizer.get_command_id('eop')
        pad_id = self.tokenizer.get_command_id('pad')
        if self.split in ['train', 'dev']:
            masked_src, masked_tgt = self.mask_text(source_text)
            source_text = masked_src

        def pad_to(text, max_len, pad_id):
            if len(text) > max_len:
                text = text[:max_len]
            else:
                text = text + [pad_id] * (max_len - len(text))
            return text

        source_tokens = self.tokenizer.EncodeAsIds(" " + source_text)
        source_tokens = pad_to(source_tokens, self.max_src_length, pad_id)
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        if self.split in ['train', 'dev']:
            mask_positions = [
                i for i, x in enumerate(source_tokens) if x == mask_id
            ]
            assert len(mask_positions) <= len(masked_tgt)
            tokens = source_tokens
            target_ids = [0] * len(source_tokens)
            loss_mask = [0] * len(source_tokens)
            for i, mask_pos in enumerate(mask_positions):
                tgt_text = masked_tgt[i]
                tgt_tokens = self.tokenizer.EncodeAsIds(" " + tgt_text)
                tokens += [sop_id] + tgt_tokens
                target_ids += tgt_tokens + [eop_id]
                loss_mask += [1] * (len(tgt_tokens) + 1)
                position_ids += [mask_pos] * (len(tgt_tokens) + 1)
                block_position_ids += [
                    i + 1 for i in range(len(tgt_tokens) + 1)
                ]
            max_length = self.max_src_length + int(
                self.max_src_length * self.args.blank_maskratio)
            tokens = pad_to(tokens, max_length, pad_id)
            target_ids = pad_to(target_ids, max_length, pad_id)
            loss_mask = pad_to(loss_mask, max_length, 0)
            position_ids = pad_to(position_ids, max_length, 0)
            block_position_ids = pad_to(block_position_ids, max_length, 0)
            position_ids = [position_ids, block_position_ids]
            sample = {
                'input_ids': np.array(tokens, dtype=np.int64),
                'target': np.array(target_ids, dtype=np.int64),
                'attention_mask': np.array(sep, dtype=np.int64),
                'loss_mask': np.array(loss_mask, dtype=np.int64),
                "position_id": np.array(position_ids, dtype=np.int64),
                "uid": example.guid
            }
        else:
            tokens = source_tokens + [sop_id]
            mask_pos = source_tokens.index(mask_id)
            position_ids = position_ids + [mask_pos]
            block_position_ids = block_position_ids + [1]
            position_ids = [position_ids, block_position_ids]
            sample = {
                'input_ids': np.array(tokens, dtype=np.int64),
                'attention_mask': np.array(sep, dtype=np.int64),
                "position_id": np.array(position_ids, dtype=np.int64),
                "uid": example.guid
            }
        return sample

    def mask_text(self, text):
        tokens = text.split()
        mask_ratio = self.args.blank_maskratio
        n = len(tokens)
        indices = sorted(self.random.sample(range(n), int(n * mask_ratio)))
        masked_src, masked_tgt = "", []
        for i, idx in enumerate(indices):
            if i == 0 or idx != indices[i - 1] + 1:
                masked_tgt.append("")
            masked_tgt[-1] += " " + tokens[idx]
            tokens[idx] = "[MASK]"
        for i, token in enumerate(tokens):
            if i != 0 and token == "[MASK]" and tokens[i - 1] == "[MASK]":
                continue
            masked_src += " " + token
        return masked_src, masked_tgt
