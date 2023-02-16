# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.utils.data
import random
import copy
import numpy as np
import math
from torch.utils.data.dataloader import default_collate
from scipy.stats import poisson
from flagai.data.dataset.data_utils import build_sample
from flagai.data.dataset.superglue.control import PVPS, SuperGlueProcessor, PROCESSOR_DICT


def rindex(lst, val, start=None):
    if start is None:
        start = len(lst) - 1
    for i in range(start, -1, -1):
        if lst[i] == val:
            return i
    return -1


def index_in_list(lst, val, start=None):
    if start is None:
        start = 0
    for i in range(start, len(lst)):
        if lst[i] == val:
            return i
    return -1


def my_collate(batch):
    new_batch = [{key: value
                  for key, value in sample.items() if key != 'uid'}
                 for sample in batch]
    text_list = [sample['input_ids'] for sample in batch]

    def pad_choice_dim(data, choice_num):
        if data is None:
            return data
        if len(data) < choice_num:
            data = np.concatenate([data] + [data[0:1]] *
                                  (choice_num - len(data)))
        return data

    if len(text_list[0].shape) == 2:
        choice_nums = list(map(len, text_list))
        max_choice_num = max(choice_nums)
        for i, sample in enumerate(new_batch):
            for key, value in sample.items():
                if key not in ['labels', 'meta']:
                    sample[key] = pad_choice_dim(value, max_choice_num)
                else:
                    sample[key] = value
            sample['loss_mask'] = np.array([1] * choice_nums[i] + [0] *
                                           (max_choice_num - choice_nums[i]),
                                           dtype=np.int64)

    if 'dec_text' in new_batch[0]:
        choice_nums = [len(sample['dec_text']) for sample in new_batch]
        if choice_nums.count(choice_nums[0]) != len(choice_nums):
            max_choice_num = max(choice_nums)
            for i, sample in enumerate(new_batch):
                for key, value in sample.items():
                    if key.startswith('dec_'):
                        sample[key] = pad_choice_dim(value, max_choice_num)
                sample['loss_mask'] = np.array(
                    [1] * choice_nums[i] + [0] *
                    (max_choice_num - choice_nums[i]),
                    dtype=np.int64)

    new_batch = default_collate(new_batch)
    if 'uid' in batch[0]:
        uid_list = [sample['uid'] for sample in batch]
        new_batch['uid'] = uid_list

    return new_batch


class ConstructSuperglueStrategy:

    def __init__(self, args, tokenizer, task_name, custom_pvp=None):
        # pattern_id, seq_length, num_prompt_tokens, multi_token, segment_length, fast_decode, dataset_type, cloze_val=True
        self.tokenizer = tokenizer
        self.cloze_eval = args.cloze_eval
        # self.processor = PROCESSOR_DICT[task_name]
        self.processor = SuperGlueProcessor().get_processor(None,
                                                            task_name)(False)
        pvp_func = custom_pvp if custom_pvp else PVPS[task_name]
        self.pvp = pvp_func(args,
                                tokenizer,
                                self.processor.get_labels(),
                                args.seq_length,
                                pattern_id=args.pattern_id,
                                num_prompt_tokens=args.num_prompt_tokens,
                                is_multi_token=args.multi_token,
                                max_segment_length=args.segment_length,
                                fast_decode=args.fast_decode)



        self.args = args

    def __call__(self, examples):
        samples = []
        for example in examples:
            if self.cloze_eval:
                sample = self.pvp.encode(example, {})
            else:
                sample = self.processor.encode(example, self.tokenizer,
                                               self.args.seq_length, self.args)
            samples.append(sample)
        return my_collate(samples)


class ConstructSeq2seqStrategy:

    def __init__(self, args, tokenizer, task_name):
        # pattern_id, seq_length, num_prompt_tokens, multi_token, segment_length, fast_decode, dataset_type, cloze_val=True
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.args = args

    def encode(self, example):
        cls_id = self.tokenizer.get_command_id('cls')
        mask_token = 'sMASK' if self.args.task_mask else 'MASK'
        mask_id = self.tokenizer.get_command_id(mask_token)
        pad_id = self.tokenizer.get_command_id('pad')
        sop_id = self.tokenizer.get_command_id('sop')
        eop_id = self.tokenizer.get_command_id('eop')
        if self.task_name in [
                "gigaword", "cnn_dm", "cnn_dm_original", "xsum", "lang8_hsk"
        ]:
            source_text, target_text = example.text_a, example.text_b
            source_tokens = self.tokenizer.EncodeAsIds(" " + source_text)
            prompt = [cls_id, mask_id
                      ] + self.tokenizer.EncodeAsIds(" Content:")
            if len(source_tokens) > self.args.max_src_length - len(prompt):
                source_tokens = source_tokens[:self.args.max_src_length -
                                              len(prompt)]
            source_tokens = prompt + source_tokens
        elif self.task_name == "squad_generation":
            source_text = example.text_a
            target_text, answer = example.meta["question"], example.meta[
                "answer"]
            source_tokens = self.tokenizer.EncodeAsIds(source_text.rstrip() +
                                                       " Question:")
            answer_tokens = self.tokenizer.EncodeAsIds(" Answer: " + answer)
            if len(source_tokens
                   ) > self.args.max_src_length - len(answer_tokens) - 2:
                max_src_length = self.args.max_src_length - len(
                    answer_tokens) - 2
                answer_pattern = self.tokenizer.EncodeAsIds(" " + answer)

                def sub_finder(mylist, pattern):
                    matches = []
                    for i in range(len(mylist)):
                        if mylist[i] == pattern[0] and mylist[
                                i:i + len(pattern)] == pattern:
                            matches.append(i)
                    return matches

                answer_indices = sub_finder(source_tokens, answer_pattern)
                if len(answer_indices) == 0:
                    print(f"Answer {answer} not exists in the source text")
                    source_tokens = source_tokens[:max_src_length]
                else:
                    start_index = max(answer_indices[0] - max_src_length // 2,
                                      0)
                    source_tokens = source_tokens[start_index:start_index +
                                                  max_src_length]
            source_tokens = [cls_id] + source_tokens + [mask_id
                                                        ] + answer_tokens
        elif self.task_name in ["cmrc"]:
            mask_id = self.tokenizer.get_command_id('MASK')
            source_text = example.text_a
            target_text = example.meta["answer"].strip()
            question = example.meta["question"].strip()
            source_tokens = self.tokenizer.EncodeAsIds(source_text.rstrip())
            question_tokens = self.tokenizer.EncodeAsIds("问题：" + question +
                                                         "答案：")
            max_src_length = self.args.max_src_length - len(
                question_tokens) - 2
            if max_src_length <= 0:
                question_tokens = question_tokens[self.args.max_src_length //
                                                  4]
            source_tokens = [cls_id] + question_tokens + [
                mask_id
            ] + source_tokens[:max_src_length]
        elif self.task_name in ["wsc"]:
            mask_id = self.tokenizer.get_command_id('MASK')
            source_text = example.text_a
            target_text = example.meta["answer"].strip()
            question = example.meta["question"].strip()
            source_tokens = self.tokenizer.EncodeAsIds(source_text.rstrip())
            question_tokens = self.tokenizer.EncodeAsIds("what does " +
                                                         question + "mean: ")
            max_src_length = self.args.max_src_length - len(
                question_tokens) - 2
            if max_src_length <= 0:
                print(question)
                question_tokens = question_tokens[self.args.max_src_length //
                                                  4]
            source_tokens = [cls_id] + question_tokens + [
                mask_id
            ] + source_tokens[:max_src_length]
        else:
            raise NotImplementedError
        if len(source_tokens) < self.args.max_src_length:
            source_tokens = source_tokens + [pad_id] * (
                self.args.max_src_length - len(source_tokens))
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        mask_pos = source_tokens.index(mask_id)

        target_tokens = self.tokenizer.EncodeAsIds(" " + target_text)
        target_tokens = target_tokens + [eop_id]
        if len(target_tokens) > self.args.max_tgt_length:
            target_tokens = target_tokens[:self.args.max_tgt_length]
        loss_mask = [1] * len(target_tokens)
        if len(target_tokens) < self.args.max_tgt_length:
            loss_mask += [0] * (self.args.max_tgt_length - len(target_tokens))
            target_tokens += [pad_id] * (self.args.max_tgt_length -
                                         len(target_tokens))
        tokens = source_tokens + [sop_id] + target_tokens[:-1]
        loss_mask = [0] * len(source_tokens) + loss_mask
        target_ids = [0] * len(source_tokens) + target_tokens
        position_ids += [mask_pos] * len(target_tokens)
        if self.args.no_block_position:
            block_position_ids += [1] * len(target_tokens)
        else:
            block_position_ids += list(range(1, len(target_tokens) + 1))
        position_ids = [position_ids, block_position_ids]
        sample = build_sample(ids=tokens,
                              positions=position_ids,
                              target=target_ids,
                              masks=sep,
                              loss_mask=loss_mask,
                              unique_id=example.guid)
        return sample

    def __call__(self, examples):
        samples = []
        for example in examples:
            sample = self.encode(example)
            samples.append(sample)
        return my_collate(samples)


class ConstructBlockStrategy:

    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 bert_prob=0.4,
                 gap_sentence_prob=0.3,
                 gpt_infill_prob=0.5,
                 gpt_min_ratio=0.25,
                 bert_ratio=0.15,
                 gap_sentence_ratio=0.15,
                 average_block_length=3,
                 max_block_length=40,
                 block_mask_prob=0.0,
                 context_mask_ratio=0.0,
                 context_mask_range=3,
                 short_seq_prob=0.0,
                 single_span_prob=0.0,
                 block_position_encoding=True,
                 encoder_decoder=False,
                 shuffle_blocks=True,
                 sentinel_token=False,
                 task_mask=False,
                 random_position=False,
                 masked_lm=False,
                 eod_token=None):
        self.eod_token = eod_token
        self.tokenizer = tokenizer
        self.count = 0
        self.max_seq_length = max_seq_length
        # self.rank = mpu.get_data_parallel_rank()
        # self.world_size = mpu.get_data_parallel_world_size()
        # self.rank = 0
        # self.world_size = 1
        assert 0.0 <= bert_prob <= 1.0
        self.bert_prob = bert_prob
        self.gap_sentence_prob = gap_sentence_prob
        self.gpt_prob = 1 - bert_prob - gap_sentence_prob
        assert self.gpt_prob >= -1e-10
        self.infill_prob = gpt_infill_prob
        self.gpt_min_ratio = gpt_min_ratio
        self.bert_ratio = bert_ratio
        self.gap_sentence_ratio = gap_sentence_ratio
        self.block_length_distribution = [
            poisson.pmf(i, average_block_length)
            for i in range(1, max_block_length)
        ]
        self.block_mask_prob = block_mask_prob
        self.context_mask_ratio = context_mask_ratio
        self.context_mask_range = context_mask_range
        self.short_seq_prob = short_seq_prob
        self.single_span_prob = single_span_prob
        self.block_position_encoding = block_position_encoding
        self.encoder_decoder = encoder_decoder
        self.shuffle_blocks = shuffle_blocks
        self.sentinel_token = sentinel_token
        self.generation_mask = 'gMASK' if task_mask else 'MASK'
        self.generation_mask = self.tokenizer.get_command_id(
            self.generation_mask)
        self.gap_sentence_mask = 'sMASK' if task_mask else 'MASK'
        self.gap_sentence_mask = self.tokenizer.get_command_id(
            self.gap_sentence_mask)
        self.random_position = random_position
        self.masked_lm = masked_lm

    def contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        if ';' in tok:
            return True
        if ':' in tok:
            return True
        if '。' in tok:
            return True
        if '？' in tok:
            return True
        if '！' in tok:
            return True
        if '；' in tok:
            return True
        if '…' in tok:
            return True
        if '\n' in tok:
            return True
        return False

    @staticmethod
    def sample_spans(span_lengths, total_length, rng, offset=0):
        blank_length = total_length - sum(span_lengths)
        m = blank_length - len(span_lengths) + 1
        places = [rng.randrange(m + 1) for _ in range(len(span_lengths))]
        places.sort()
        spans = []
        for place, span_length in zip(places, span_lengths):
            start = offset + place
            end = offset + place + span_length
            spans.append((start, end))
            offset += span_length + 1
        return spans

    def sample_span_in_document(self, tokens, masked_lengths, rng):
        rng.shuffle(masked_lengths)
        mask_spans = []
        mask_index = 0
        indices = [-1] + np.where(tokens == self.eod_token)[0].tolist()
        last_index = len(tokens)
        documents = []
        for index in reversed(indices):
            start_index = index
            if start_index + 1 < len(tokens) and tokens[
                    start_index + 1] == self.tokenizer.get_command_id('cls'):
                start_index += 1
            length = last_index - start_index - 1
            if last_index == len(tokens) and length > 0:
                length -= 1
            documents.append((start_index + 1, length))
            last_index = index
        documents.sort(key=lambda x: x[1])
        for i, (offset, length) in enumerate(documents):
            if i == len(documents) - 1:
                current_masked_length, current_count = 0, 0
                while mask_index + current_count < len(
                        masked_lengths
                ) and masked_lengths[
                        mask_index +
                        current_count] + current_masked_length + current_count <= length:
                    current_masked_length += masked_lengths[mask_index +
                                                            current_count]
                    current_count += 1
                if current_count > 0:
                    spans = self.sample_spans(
                        masked_lengths[mask_index:mask_index + current_count],
                        length,
                        rng,
                        offset=offset)
                    mask_spans += spans
                if mask_index + current_count < len(masked_lengths) - 1:
                    print(length, masked_lengths[mask_index:],
                          masked_lengths[:mask_index], indices)
            else:
                current_masked_total = int(length * self.bert_ratio)
                current_masked_length, current_count = 0, 0
                while mask_index + current_count < len(
                        masked_lengths
                ) and masked_lengths[
                        mask_index +
                        current_count] + current_masked_length <= current_masked_total:
                    current_masked_length += masked_lengths[mask_index +
                                                            current_count]
                    current_count += 1
                if current_count > 0:
                    spans = self.sample_spans(
                        masked_lengths[mask_index:mask_index + current_count],
                        length,
                        rng,
                        offset=offset)
                    mask_spans += spans
                    mask_index += current_count
        return mask_spans

    def make_masked_data(self,
                         tokens,
                         loss_masks,
                         attention_mask,
                         block_spans,
                         rng,
                         task='bert'):

        position_ids = np.arange(len(tokens), dtype=np.int64)
        targets = copy.deepcopy(tokens)
        mask_id = self.tokenizer.get_command_id('MASK')
        mlm_masks = np.zeros(len(tokens), dtype=np.int64)
        for start, end in block_spans:
            for idx in range(start, end):
                tokens[idx] = mask_id
            mlm_masks[start:end] = 1
        loss_masks = loss_masks * mlm_masks
        return tokens, targets, loss_masks, position_ids

    def make_block_data(self,
                        tokens,
                        loss_masks,
                        attention_mask,
                        block_spans,
                        rng,
                        task='bert'):
        text_length = len(tokens)
        position_ids = np.ones(len(tokens), dtype=np.int64)
        for start, end in block_spans:
            position_ids[start + 1:end] = 0
        position_ids = np.cumsum(position_ids) - 1
        if self.random_position and position_ids[-1] < self.max_seq_length - 1:
            position_bias = self.max_seq_length - position_ids[-1]
            position_bias = rng.randrange(0, position_bias)
            position_ids = position_ids + position_bias
        if self.encoder_decoder or not self.shuffle_blocks:
            block_spans.sort(key=lambda x: x[0])
        else:
            rng.shuffle(block_spans)
        if self.sentinel_token:
            block_spans = [(start, end, idx)
                           for idx, (start, end) in enumerate(block_spans)]
        else:
            block_spans = [(start, end, 0) for start, end in block_spans]
        target_tokens, target_position_ids, target_block_position_ids, targets = [], [], [], []
        for start, end, idx in block_spans:
            sop_token = 'sop' if idx == 0 else f"sop{idx}"
            target_tokens.append([self.tokenizer.get_command_id(sop_token)])
            span_tokens = copy.deepcopy(tokens[start:end])
            if self.block_mask_prob > 0.0 and task == 'bert':
                for sub_idx in range(len(span_tokens)):
                    if random.random() < self.block_mask_prob:
                        span_tokens[sub_idx] = self.tokenizer.get_command_id(
                            'dBLOCK')
            target_tokens.append(span_tokens)
            targets.append(tokens[start:end])
            targets.append([self.tokenizer.get_command_id('eop')])
            if not self.sentinel_token:
                target_position_id = position_ids[start:end]
                target_position_ids.append(target_position_id)
                target_position_ids.append([target_position_id[0]])
            else:
                target_position_ids.append([self.max_seq_length] *
                                           (end - start + 1))
            if self.block_position_encoding:
                target_block_position_ids.append(
                    np.arange(1, end - start + 2, dtype=np.int64))
            else:
                target_block_position_ids.append([1] * (end - start + 1))
        block_spans.sort(key=lambda x: x[0])
        source_tokens, source_position_ids, local_spans = [], [], []
        last, current_length = 0, 0
        for start, end, idx in block_spans:
            if task == 'generation':
                mask_id = self.generation_mask
            elif task == 'gap_sentence':
                mask_id = self.gap_sentence_mask
            else:
                mask_token = 'MASK' if idx == 0 else f'MASK{idx}'
                mask_id = self.tokenizer.get_command_id(mask_token)
            local_spans.append((current_length, current_length + start - last))
            source_tokens.append(tokens[last:start])
            source_tokens.append([mask_id])
            source_position_ids.append(position_ids[last:start])
            source_position_ids.append([position_ids[start]])
            current_length += start - last + 1
            last = end
        if last < len(tokens):
            local_spans.append(
                (current_length, current_length + len(tokens) - last))
            source_tokens.append(tokens[last:])
            source_position_ids.append(position_ids[last:])
        source_length = sum(map(len, source_tokens))
        if attention_mask is not None:
            assert source_length == attention_mask
        if target_tokens and self.eod_token in np.concatenate(
                target_tokens).tolist():
            print("Found EOS in target", self.tokenizer.DecodeIds(tokens))
            raise RuntimeError
        if self.encoder_decoder:
            target_tokens = target_tokens + [
                self.tokenizer.get_command_id('eop')
            ]
            loss_masks = np.ones(len(target_tokens), dtype=np.int64)
            return source_tokens, target_tokens, loss_masks
        else:
            tokens = np.concatenate(source_tokens + target_tokens)
            if task == 'bert' and self.context_mask_ratio > 0:
                mask_candidates = set()
                for start, end in local_spans:
                    if start != 0:
                        local_end = min(end, start + self.context_mask_range)
                        mask_candidates.update(range(start, local_end))
                    if end != 0:
                        local_start = max(start, end - self.context_mask_range)
                        mask_candidates.update(range(local_start, end))
                mask_pos = rng.sample(
                    mask_candidates,
                    int(self.context_mask_ratio * text_length))
                for pos in mask_pos:
                    tokens[pos] = self.tokenizer.get_command_id('dBLOCK')
            targets = np.concatenate(source_tokens + targets)
            loss_masks = np.ones(len(tokens), dtype=np.int64)
            loss_masks[:source_length] = 0
            position_ids = np.concatenate(source_position_ids +
                                          target_position_ids)
            block_position_ids = np.concatenate(
                [np.zeros(source_length, dtype=np.int64)] +
                target_block_position_ids)
            position_ids = np.stack([position_ids, block_position_ids], axis=0)
            if attention_mask is not None:
                return tokens, targets, loss_masks, position_ids
            else:
                return tokens, targets, loss_masks, position_ids, source_length

    def generate_blank_data(self,
                            sample,
                            masked_lengths,
                            attention_mask,
                            rng,
                            task='bert'):
        rng.shuffle(masked_lengths)
        tokens, loss_masks = sample['input_ids'], sample['loss_mask']
        assert tokens[0] == self.tokenizer.get_command_id('cls')
        block_spans = self.sample_span_in_document(tokens, masked_lengths, rng)
        if len(block_spans) < len(masked_lengths):
            return None
        if self.masked_lm:
            data = self.make_masked_data(tokens, loss_masks, attention_mask,
                                         block_spans, rng)
        else:
            data = self.make_block_data(tokens,
                                        loss_masks,
                                        attention_mask,
                                        block_spans,
                                        rng,
                                        task=task)
        return data

    def split_samples(self, samples, rng):
        target_length = rng.randrange(32, self.max_seq_length - 1)
        num_splits = (self.max_seq_length - 1) // target_length
        new_samples = []
        cls_id = self.tokenizer.get_command_id('cls')
        eos_id = self.tokenizer.get_command_id('eos')
        for sample in samples:
            tokens, loss_masks = sample['input_ids'][1:], sample['loss_mask'][
                1:]
            for _ in range(num_splits):
                if target_length >= len(tokens):
                    new_tokens, new_loss_masks = tokens, loss_masks
                else:
                    random_start = rng.randrange(0,
                                                 len(tokens) - target_length)
                    while random_start > 0 and (
                            tokens[random_start] == eos_id
                            or not (self.contains_sentence_end(
                                tokens[random_start - 1])
                                    or tokens[random_start - 1] == eos_id)):
                        random_start -= 1
                    random_end = random_start + target_length
                    while random_end > random_start and not (
                            self.contains_sentence_end(tokens[random_end - 1])
                            or tokens[random_end - 1] == eos_id):
                        random_end -= 1
                    if random_end - random_start < target_length // 2:
                        random_end = random_start + target_length
                    new_tokens, new_loss_masks = tokens[
                        random_start:random_end], loss_masks[
                            random_start:random_end]
                new_tokens = np.concatenate(([cls_id], new_tokens))
                new_loss_masks = np.concatenate(([0], new_loss_masks))
                new_samples.append({
                    'input_ids': new_tokens,
                    'loss_mask': new_loss_masks
                })
        return new_samples

    def __call__(self, samples):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        rng = random.Random((self.count * num_workers + worker_id))
        self.count += 1
        token_batch, target_batch, loss_mask_batch, position_id_batch = [], [], [], []
        source_batch, target_batch = [], []
        if rng.random() < self.short_seq_prob:
            samples = self.split_samples(samples, rng)
        rand = rng.random()
        single_span = rand < self.single_span_prob
        rand = 0.0 if single_span else rng.random()
        attention_mask = []
        if rand < self.bert_prob:
            mode = 'bert'
            for sample in samples:
                if single_span:
                    masked_lengths = [
                        rng.choices(range(
                            1,
                            len(self.block_length_distribution) + 1),
                                    weights=self.block_length_distribution)[0]
                    ]
                    masked_count = masked_lengths[0]
                else:
                    masked_lengths, masked_count = [], 0
                    while masked_count < int(
                            self.bert_ratio * len(sample['input_ids'])):
                        block_length = rng.choices(
                            range(1,
                                  len(self.block_length_distribution) + 1),
                            weights=self.block_length_distribution)[0]
                        masked_lengths.append(block_length)
                        masked_count += block_length
                if self.masked_lm:
                    sep = len(sample['input_ids'])
                else:
                    sep = len(sample['input_ids']) - masked_count + len(
                        masked_lengths)
                data = self.generate_blank_data(sample,
                                                masked_lengths,
                                                sep,
                                                rng,
                                                task='bert')
                if data is not None:
                    if self.encoder_decoder:
                        source_tokens, target_tokens, loss_masks = data
                        source_batch.append(source_tokens)
                        target_batch.append(target_tokens)
                        loss_mask_batch.append(loss_masks)
                    else:
                        tokens, targets, loss_masks, position_ids = data
                        token_batch.append(tokens)
                        target_batch.append(targets)
                        loss_mask_batch.append(loss_masks)
                        position_id_batch.append(position_ids)
                    attention_mask.append(sep)

        elif rand < self.bert_prob + self.gap_sentence_prob:
            mode = 'sentence'
            for sample in samples:
                tokens, loss_masks = sample['input_ids'], sample['loss_mask']
                sentence_spans = []
                last_index = 1 if tokens[0] == self.tokenizer.get_command_id(
                    'cls') else 0
                for i in range(len(tokens)):
                    if self.contains_sentence_end(tokens[i]):
                        if last_index < i + 1:
                            sentence_spans.append((last_index, i + 1))
                        last_index = i + 1
                    elif tokens[i] == self.tokenizer.get_command_id('eos'):
                        last_index = i + 1
                if last_index < len(tokens):
                    sentence_spans.append((last_index, len(tokens)))
                if not sentence_spans and torch.distributed.get_rank() == 0:
                    try:
                        print(self.tokenizer.DecodeIds(tokens[1:]))
                    except IndexError:
                        print(tokens[1:])
                rng.shuffle(sentence_spans)
                block_spans, block_length = [], 0
                for start, end in sentence_spans:
                    block_spans.append((start, end))
                    block_length += end - start
                    if block_length >= int(
                            self.gap_sentence_ratio * len(tokens)):
                        break
                data = self.make_block_data(tokens,
                                            loss_masks,
                                            None,
                                            block_spans,
                                            rng,
                                            task='gap_sentence')
                tokens, targets, loss_masks, position_ids, sep = data
                token_batch.append(tokens)
                target_batch.append(targets)
                loss_mask_batch.append(loss_masks)
                position_id_batch.append(position_ids)
                attention_mask.append(sep)
        else:
            mode = 'gpt'
            max_generation_length = rng.randint(
                int(self.gpt_min_ratio *
                    min(map(lambda x: len(x['input_ids']), samples))),
                max(map(lambda x: len(x['input_ids']), samples)) - 2)
            for sample in samples:
                generation_length = min(max_generation_length,
                                        len(sample['input_ids']) - 2)
                attention_mask.append(
                    len(sample['input_ids']) - generation_length + 1)
                multiple_doc = index_in_list(
                    sample['input_ids'],
                    self.tokenizer.get_command_id('eos')) not in [
                        -1, len(sample['input_ids']) - 1
                    ]
                if multiple_doc or rng.random() < self.infill_prob:
                    division = len(sample['input_ids']) - generation_length
                    tokens, loss_masks = sample['input_ids'], sample[
                        'loss_mask']
                    source_tokens, target_tokens = tokens[:division], tokens[
                        division:]
                    target_masks = loss_masks[division:]
                    tokens = np.concatenate((source_tokens, [
                        self.generation_mask,
                        self.tokenizer.get_command_id('sop')
                    ], target_tokens[:-1]))
                    targets = np.concatenate(
                        (source_tokens, [self.generation_mask], target_tokens))
                    loss_masks = np.concatenate(
                        (np.zeros(len(source_tokens) + 1,
                                  dtype=np.int64), target_masks))
                    token_batch.append(tokens)
                    target_batch.append(targets)
                    loss_mask_batch.append(loss_masks)
                    position_ids = np.arange(len(source_tokens) +
                                             len(target_tokens) + 1,
                                             dtype=np.int64)
                    position_ids[len(source_tokens) + 1:] = len(source_tokens)
                    if self.block_position_encoding:
                        block_position_ids = np.concatenate(
                            (np.zeros(len(source_tokens), dtype=np.int64),
                             np.arange(len(target_tokens) + 1,
                                       dtype=np.int64)))
                    else:
                        block_position_ids = np.concatenate(
                            (np.zeros(len(source_tokens) + 1, dtype=np.int64),
                             np.ones(len(target_tokens) + 1, dtype=np.int64)))
                    position_id_batch.append(
                        np.stack([position_ids, block_position_ids], axis=0))
                else:
                    tokens, targets, loss_masks, position_ids = self.generate_blank_data(
                        sample, [generation_length],
                        attention_mask[-1],
                        rng,
                        task='generation')
                    token_batch.append(tokens)
                    target_batch.append(targets)
                    loss_mask_batch.append(loss_masks)
                    position_id_batch.append(position_ids)
                    if tokens is None:
                        print(sample, generation_length, multiple_doc)
        if self.encoder_decoder:
            return {
                'input_ids': torch.tensor(source_batch, dtype=torch.long),
                'labels': torch.tensor(target_batch, dtype=torch.long),
                'loss_mask': torch.tensor(loss_mask_batch, dtype=torch.long)
            }
        else:
            token_batch, target_batch, loss_mask_batch, position_id_batch = self.pad_batch(
                token_batch, target_batch, loss_mask_batch, position_id_batch)
            return {
                'input_ids': torch.tensor(token_batch, dtype=torch.long),
                'target_ids': torch.tensor(target_batch, dtype=torch.long),
                'loss_mask': torch.tensor(loss_mask_batch, dtype=torch.long),
                'position_ids': torch.tensor(position_id_batch,
                                             dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask,
                                               dtype=torch.long),
                'mode': mode
            }

    @staticmethod
    def pad_batch(token_batch, target_batch, loss_mask_batch,
                  position_id_batch):
        seq_lengths = list(map(len, token_batch))
        if seq_lengths.count(seq_lengths[0]) != len(seq_lengths):
            max_length = max(seq_lengths)
            token_batch = [
                np.concatenate(
                    (tokens, np.zeros(max_length - len(tokens),
                                      dtype=np.int64)))
                for tokens in token_batch
            ]
            target_batch = [
                np.concatenate((targets,
                                np.zeros(max_length - len(targets),
                                         dtype=np.int64)))
                for targets in target_batch
            ]
            loss_mask_batch = [
                np.concatenate((loss_masks,
                                np.zeros(max_length - len(loss_masks),
                                         dtype=np.int64)))
                for loss_masks in loss_mask_batch
            ]
            position_id_batch = [
                np.concatenate((position_ids,
                                np.zeros(
                                    (2, max_length - position_ids.shape[1]),
                                    dtype=np.int64)),
                               axis=1) for position_ids in position_id_batch
            ]
        return token_batch, target_batch, loss_mask_batch, position_id_batch
