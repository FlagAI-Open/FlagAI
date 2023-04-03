# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI GPT."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
from io import open
import sentencepiece as spm
import jieba

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func


class CPMTokenizer(object):

    def __init__(self, vocab_file, model_file, max_length=None):
        self.max_len = max_length if max_length is not None else int(1e12)
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.sp_model = spm.SentencePieceProcessor(model_file=model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")
        self.token_start_id = 0
        self.token_end_id = 3
        self.token_unk_id = 0
        self.eod_id = self.encoder['<eod>']

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        vocab = {
            self.convert_id_to_token(i): i
            for i in range(self.vocab_size)
        }
        return vocab
    
    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    @property
    def eod(self):
        return self.eod_id

    def tokenize(self, text):
        """ Tokenize a string. """
        seg_list = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        new_seg = "".join(seg_list)
        return self.sp_model.encode(new_seg)

    def encode(self, text):
        res = self.tokenize(text)
        return res
    
    def convert_tokens_to_ids(self, tokens):
        return [self.sp_model.PieceToId(token) for token in tokens]

    def convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, idx):
        return self.sp_model.IdToPiece(int(idx))

    def convert_ids_to_tokens(self, idxs):
        return [self.sp_model.IdToPiece(int(idx)) for idx in idxs]

    def decode(self, tokens):
        text = self.sp_model.decode(tokens)
        text = text.replace(' ', '').replace('\u2582',
                                             ' ').replace('\u2583', '\n')
        return text

    def encode_plus(self, text, max_length=None):
        res = self.encode(text)

        return {"input_ids": res}
    
    def convert_tokens_to_string(self, tokens, all_command_token={}):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in all_command_token:
                out_string += self.sp_model.decode_pieces(
                    current_sub_tokens) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode_pieces(current_sub_tokens)
        return out_string.strip()
