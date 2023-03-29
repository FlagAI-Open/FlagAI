# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from pkgutil import get_loader
from torch.utils.data import Dataset
from itertools import accumulate
from bisect import bisect_right
import random
import numpy as np
import torch
import os
from flagai.logger import log_dist

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
TRUE_DEV_SET = "true_dev"
UNLABELED_SET = "unlabeled"

SPLIT_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, TRUE_DEV_SET, UNLABELED_SET]


class BlockDataset(Dataset):

    def __init__(self,
                 ds,
                 tokenizer,
                 max_seq_len=1024,
                 sample_across_doc=True,
                 non_sentence_start=0.0,
                 filter_english=False):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.ds_len = len(self.ds)
        self.num_samples = 1000 * self.ds_len
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.sample_across_doc = sample_across_doc
        self.non_sentence_start = non_sentence_start
        self.filter_english = filter_english
        self.weighting, self.total_len = None, None
        self.is_lazy = False
        # if self.filter_english:
        #     import fasttext
        #     self.model = fasttext.load_model('/mnt/lid.176.bin')
        #     if torch.cuda.is_available():
        #         print_rank_0("Load language detection model")
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_weighting()

    def init_weighting(self):
        if self.is_lazy:
            lens = np.array(
                [self.ds.get_text_len(idx) for idx in range(len(self.ds))])
        else:
            lens = np.array([
                len(d['text']) if isinstance(d, dict) else len(d)
                for d in self.ds
            ])
        self.total_len = np.sum(lens)
        if torch.cuda.is_available():
            log_dist(
                f"Dataset document count {len(lens)}, token count {self.total_len}, non sentence start{self.non_sentence_start}"
            )
        self.weighting = list(accumulate(lens))

    def get_weighted_samples(self, np_rng):
        while True:
            idx = np_rng.randint(self.total_len)
            data_idx = bisect_right(self.weighting, idx)
            tokens, loss_mask = self.getidx(data_idx)
            if self.filter_english:
                text = self.tokenizer.DecodeIds(tokens[:1024])
                lang = self.model.predict(text.replace('\n', ''))[0][0]
                if lang == '__label__en':
                    break
            else:
                break
        return tokens, loss_mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(
            seed=[rng.randint(0, 2**32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        tokens, loss_mask = self.get_weighted_samples(rng)
        # truncate or pad tokens
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self.max_seq_len + 1

        # randomly choose a position for start
        if tokens_to_strip > 0:
            move_count = 0
            strip_left_tokens = rng.randint(tokens_to_strip)
            if rng.random() > self.non_sentence_start:
                if rng.random() < 0.5:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens > 0 and not self.contains_sentence_end(
                            tokens[strip_left_tokens - 1]):
                        strip_left_tokens -= 1
                        move_count += 1
                else:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens < len(
                            tokens) and not self.contains_sentence_end(
                                tokens[strip_left_tokens - 1]):
                        strip_left_tokens += 1
                        move_count += 1
            tokens = [self.tokenizer.get_command_id('cls')
                      ] + tokens[strip_left_tokens:]
            loss_mask = [0] + loss_mask[strip_left_tokens:]
            if len(tokens) == 2 and tokens[1] == self.tokenizer.get_command_id(
                    'eos'):
                tokens, loss_mask = [], []
            tokens, loss_mask = self.right_strip_seq(tokens, loss_mask,
                                                     self.max_seq_len)
        else:
            tokens = [self.tokenizer.get_command_id('cls')] + tokens
            loss_mask = [0] + loss_mask
            # Sample multiple documents
            if self.sample_across_doc:
                while len(tokens) < self.max_seq_len:
                    new_tokens, new_loss_mask = self.get_weighted_samples(rng)
                    new_tokens = [self.tokenizer.get_command_id('cls')
                                  ] + new_tokens
                    new_loss_mask = [0] + new_loss_mask
                    is_last = len(new_tokens) >= self.max_seq_len - len(tokens)
                    new_tokens, new_loss_mask = self.right_strip_seq(
                        new_tokens, new_loss_mask,
                        self.max_seq_len - len(tokens))
                    tokens += new_tokens
                    loss_mask += new_loss_mask
                    if is_last:
                        break
        return {
            'input_ids': np.array(tokens),
            "loss_mask": np.array(loss_mask)
        }

    def right_strip_seq(self, tokens, loss_mask, seq_length):
        strip_right_tokens = len(tokens) - seq_length
        if strip_right_tokens > 0:
            while strip_right_tokens < len(
                    tokens) - 1 and not self.contains_sentence_end(
                        tokens[-strip_right_tokens - 1]):
                strip_right_tokens += 1
            if len(tokens) - strip_right_tokens < seq_length // 2:
                strip_right_tokens = len(tokens) - seq_length
            tokens = tokens[:-strip_right_tokens]
            loss_mask = loss_mask[:-strip_right_tokens]
        return tokens, loss_mask

    def getidx(self, data_idx):
        data = self.ds[data_idx]
        tokens, loss_masks = data['tokens'], data['loss_masks']
        tokens = tokens + [self.tokenizer.get_command_id('eos')]
        loss_masks = loss_masks + [1]
        return tokens, loss_masks

    def pad_seq(self, seq, pad_id=None):
        total_tokens = self.max_seq_len
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [
            self.tokenizer.get_command_id('pad') if pad_id is None else pad_id
        ] * (num_pad_tokens)
        return seq

    # TODO: rewrite this function for chinese
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
        if '\n' in tok:
            return True
        return False
