# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""Utilities for using and training tokenizers (char, wordpiece, sentencepiece)"""
# from collections import namedtuple
# import itertools

import logging
import re
import json
logger = logging.getLogger(__name__)
from flagai.data.tokenizer.glm_10b_en.glm_10b_en_tokenizer import bytes_to_unicode, get_pairs
import sys


class BPETokenizer(object):
    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 max_len=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.special_tokens = {}
        self.special_tokens_decoder = {}
        # self.set_special_tokens(special_tokens)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder)

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[
                        i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if sys.version_info[0] == 2:
                token = ''.join(self.byte_encoder[ord(b)] for b in token)
            else:
                token = ''.join(self.byte_encoder[b]
                                for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token
                              for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def convert_token_to_id(self, token):
        """ Converts a sequence of tokens into ids using the vocab. """
        return self.encoder.get(token, 0)

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        for token in tokens:
            ids.append(self.convert_token_to_id(token))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".
                format(len(ids), self.max_len))
        return ids

    def convert_id_to_token(self, id):
        """Converts a sequence of ids in BPE tokens using the vocab."""
        return self.decoder[id]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in BPE tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.decoder[i])
        return tokens

    def encode(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, ids):
        text = ''.join([self.decoder[id] for id in ids])
        text = bytearray([self.byte_decoder[c]
                          for c in text]).decode('utf-8', errors=self.errors)
        return text

    def convert_tokens_to_string(self, tokens, all_command_token={}):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    # def save_vocabulary(self, vocab_path):
    #     """Save the tokenizer vocabulary and merge files to a directory."""
    #     if not os.path.isdir(vocab_path):
    #         logger.error("Vocabulary path ({}) should be a directory".format(
    #             vocab_path))
    #         return
    #     vocab_file = os.path.join(vocab_path, VOCAB_NAME)
    #     merge_file = os.path.join(vocab_path, MERGES_NAME)
    #     special_tokens_file = os.path.join(vocab_path, SPECIAL_TOKENS_NAME)
    #
    #     with open(vocab_file, 'w', encoding='utf-8') as f:
    #         f.write(json.dumps(self.encoder, ensure_ascii=False))
    #
    #     index = 0
    #     with open(merge_file, "w", encoding="utf-8") as writer:
    #         writer.write(u'#version: 0.2\n')
    #         for bpe_tokens, token_index in sorted(self.bpe_ranks.items(),
    #                                               key=lambda kv: kv[1]):
    #             if index != token_index:
    #                 logger.warning(
    #                     "Saving vocabulary to {}: BPE merge indices are not consecutive."
    #                     " Please check that the tokenizer is not corrupted!".
    #                     format(merge_file))
    #                 index = token_index
    #             writer.write(' '.join(bpe_tokens) + u'\n')
    #             index += 1
    #
    #     index = len(self.encoder)
    #     with open(special_tokens_file, 'w', encoding='utf-8') as writer:
    #         for token, token_index in sorted(self.special_tokens.items(),
    #                                          key=lambda kv: kv[1]):
    #             if index != token_index:
    #                 logger.warning(
    #                     "Saving special tokens vocabulary to {}: BPE indices are not consecutive."
    #                     " Please check that the tokenizer is not corrupted!".
    #                     format(special_tokens_file))
    #                 index = token_index
    #             writer.write(token + u'\n')
    #             index += 1
    #
    #     return vocab_file, merge_file, special_tokens_file


