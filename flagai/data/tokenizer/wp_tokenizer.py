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
from collections import namedtuple
import itertools



import logging
logger = logging.getLogger(__name__)
import os
from flagai.model.file_utils import _get_model_id, _get_vocab_path
from flagai.data.tokenizer.glm_large_ch.glm_large_ch import get_encoder
from flagai.data.tokenizer.glm_10b_en.glm_10b_en_tokenizer import bytes_to_unicode
from flagai.data.tokenizer.glm_large_en.wordpiece import load_vocab, BasicTokenizer, WordpieceTokenizer
import collections
import json
import re


import logging
logger = logging.getLogger(__name__)
import os
from flagai.model.file_utils import _get_model_id, _get_vocab_path
from flagai.data.tokenizer.glm_large_ch.glm_large_ch import get_encoder
from flagai.data.tokenizer.glm_10b_en.glm_10b_en_tokenizer import bytes_to_unicode
from flagai.data.tokenizer.glm_large_en.wordpiece import load_vocab, BasicTokenizer, WordpieceTokenizer
import collections
import json
import re


class BaseTokenizer(object):
    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        cache_dir=None,
                        *inputs,
                        **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        vocab_file = 'vocab.txt'
        merges_file = 'merges.txt'
        sp_file = 'spm.model'
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), 'vocabs')
        tokenizer_class = "wp"
        # search the cache directory for certain files
        if os.path.exists(cache_dir):
            if os.path.exists(cache_dir + '/' + vocab_file):  # Temporary if statement
                if os.path.exists(cache_dir + '/' + merges_file):  # Temporary if statement
                    tokenizer_class = "bpe"
                else:
                    tokenizer_class = "wp"
            elif os.path.exists(cache_dir + '/' + sp_file):
                tokenizer_class = "sp"
        else:
            model_id = _get_model_id(pretrained_model_name_or_path)
            try:
                _get_vocab_path(cache_dir + '/', vocab_file, model_id, rank=0)
                try:
                    _get_vocab_path(cache_dir + '/', merges_file, model_id, rank=0)
                    tokenizer_class = "bpe"
                except:
                    tokenizer_class = 'wp'
            except:
                try:
                    _get_vocab_path(cache_dir + '/', sp_file, model_id, rank=0)
                    tokenizer_class = "sp"
                except:
                    raise("Error")
        resolved_vocab_file = os.path.join(cache_dir, vocab_file)
        resolved_merges_file = os.path.join(cache_dir, merges_file)
        resolved_sp_file = os.path.join(cache_dir, sp_file)
        if tokenizer_class == "wp":
            return cls._from_pretrained(resolved_vocab_file, tokenizer_class, *inputs, **kwargs)
        elif tokenizer_class == "bpe":
            return cls._from_pretrained(resolved_vocab_file, resolved_merges_file, tokenizer_class, *inputs, **kwargs)
        elif tokenizer_class == "sp":
            return get_encoder(resolved_sp_file, "")

    def __init__(self):
        self.test = 1

    def _from_pretrained(self, vocab_file=None, do_basic_tokenize=True,
                            do_lower_case=True, max_len=None,
                            never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        raise NotImplementedError(
            'TextTokenizer tokens property not implemented')

class WordpieceTokenizer(BaseTokenizer):
    def _from_pretrained(self, vocab_file=None, do_basic_tokenize=True,
                         do_lower_case=True, max_len=None,
                         never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                    .format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                  never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
        self.tokenizer_class = "wp"

    def __init__(self, name, age):
        self.name = name
        self.age = age














        # if not os.path.isfile(vocab_file):
        #     raise ValueError(
        #         "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
        #         "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
        #         .format(vocab_file))
        # self.vocab = load_vocab(vocab_file)
        # self.ids_to_tokens = collections.OrderedDict([
        #     (ids, tok) for tok, ids in self.vocab.items()
        # ])
        # self.do_basic_tokenize = do_basic_tokenize
        # if do_basic_tokenize:
        #     self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
        #                                           never_split=never_split)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        # self.max_len = max_len if max_len is not None else int(1e12)
        # self.tokenizer_class = "wp"
    #
    # def set_special_tokens(self, special_tokens):
    #     """ Add a list of additional tokens to the encoder.
    #         The additional tokens are indexed starting from the last index of the
    #         current vocabulary in the order of the `special_tokens` list.
    #     """
    #     if not special_tokens:
    #         self.special_tokens = {}
    #         self.special_tokens_decoder = {}
    #         return
    #     self.special_tokens = dict((tok, len(self.encoder) + i)
    #                                for i, tok in enumerate(special_tokens))
    #     self.special_tokens_decoder = {
    #         v: k
    #         for k, v in self.special_tokens.items()
    #     }
    #     logger.info("Special tokens {}".format(self.special_tokens))
    #
    # def _from_pretrained_bpe(self,
    #              vocab_file,
    #              merges_file,
    #              errors='replace',
    #              special_tokens=None,
    #              max_len=None):
    #     self.max_len = max_len if max_len is not None else int(1e12)
    #     self.encoder = json.load(open(vocab_file))
    #     self.decoder = {v: k for k, v in self.encoder.items()}
    #     self.errors = errors  # how to handle errors in decoding
    #     self.byte_encoder = bytes_to_unicode()
    #     self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    #     bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
    #     bpe_merges = [tuple(merge.split()) for merge in bpe_data]
    #     self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
    #     self.cache = {}
    #
    #     # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
    #     self.pat = re.compile(
    #         r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #     )
    #
    #     self.special_tokens = {}
    #     self.special_tokens_decoder = {}
    #     self.set_special_tokens(special_tokens)
    #     self.tokenizer_class = "bpe"
    #
    #
    # def tokenize(self, text):
    #     if self.do_basic_tokenize:
    #         split_tokens = []
    #         for token in self.basic_tokenizer.tokenize(text):
    #             for sub_token in self.wordpiece_tokenizer.tokenize(token):
    #                 split_tokens.append(sub_token)
    #     else:
    #         split_tokens = self.wordpiece_tokenizer.tokenize(text)
    #     return split_tokens
    #
    # def convert_tokens_to_ids(self, tokens):
    #     """Converts a sequence of tokens into ids using the vocab."""
    #     ids = []
    #     for token in tokens:
    #         ids.append(self.vocab[token])
    #     if len(ids) > self.max_len:
    #         logger.warning(
    #             "Token indices sequence length is longer than the specified maximum "
    #             " sequence length for this BERT model ({} > {}). Running this"
    #             " sequence through BERT will result in indexing errors".format(
    #                 len(ids), self.max_len))
    #     return ids
    #
    # def convert_ids_to_tokens(self, ids):
    #     """Converts a sequence of ids in wordpiece tokens using the vocab."""
    #     tokens = []
    #     for i in ids:
    #         tokens.append(self.ids_to_tokens[i])
    #     return tokens


# from flagai.data.tokenizer.tokenizer import BasicTokenizer


# class BaseTokenizer(object):
#     @classmethod
#     def from_pretrained(cls,
#                         pretrained_model_name_or_path,
#                         cache_dir=None,
#                         *inputs,
#                         **kwargs):
#         """
#         Instantiate a PreTrainedBertModel from a pre-trained model file.
#         Download and cache the pre-trained model file if needed.
#         """
#         vocab_file = 'vocab.txt'
#         merges_file = 'merges.txt'
#         sp_file = 'spm.model'
#         if cache_dir is None:
#             cache_dir = os.path.join(os.path.dirname(__file__), 'vocabs')
#         tokenizer_class = "wp"
#         # search the cache directory for certain files
#         if os.path.exists(cache_dir):
#             if os.path.exists(cache_dir + '/' + vocab_file):  # Temporary if statement
#                 if os.path.exists(cache_dir + '/' + merges_file):  # Temporary if statement
#                     tokenizer_class = "bpe"
#                 else:
#                     tokenizer_class = "wp"
#             elif os.path.exists(cache_dir + '/' + sp_file):
#                 tokenizer_class = "sp"
#         else:
#             model_id = _get_model_id(pretrained_model_name_or_path)
#             try:
#                 _get_vocab_path(cache_dir + '/', vocab_file, model_id, rank=0)
#                 try:
#                     _get_vocab_path(cache_dir + '/', merges_file, model_id, rank=0)
#                     tokenizer_class = "bpe"
#                 except:
#                     tokenizer_class = 'wp'
#             except:
#                 try:
#                     _get_vocab_path(cache_dir + '/', sp_file, model_id, rank=0)
#                     tokenizer_class = "sp"
#                 except:
#                     raise("Error")
#         resolved_vocab_file = os.path.join(cache_dir, vocab_file)
#         resolved_merges_file = os.path.join(cache_dir, merges_file)
#         resolved_sp_file = os.path.join(cache_dir, sp_file)
#         if tokenizer_class == "wp":
#             return cls._from_pretrained_wp(resolved_vocab_file, tokenizer_class, *inputs, **kwargs)
#         elif tokenizer_class == "bpe":
#             return cls._from_pretrained(resolved_vocab_file, resolved_merges_file, tokenizer_class, *inputs, **kwargs)
#         elif tokenizer_class == "sp":
#             return get_encoder(resolved_sp_file, "")
#
#     def _from_pretrained_wp(self, vocab_file=None, do_basic_tokenize=True,
#                             do_lower_case=True, max_len=None,
#                             never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
#         if not os.path.isfile(vocab_file):
#             raise ValueError(
#                 "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
#                 "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
#                 .format(vocab_file))
#         self.vocab = load_vocab(vocab_file)
#         self.ids_to_tokens = collections.OrderedDict([
#             (ids, tok) for tok, ids in self.vocab.items()
#         ])
#         self.do_basic_tokenize = do_basic_tokenize
#         if do_basic_tokenize:
#             self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
#                                                   never_split=never_split)
#         self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
#         self.max_len = max_len if max_len is not None else int(1e12)
#         self.tokenizer_class = "wp"
#
#     def set_special_tokens(self, special_tokens):
#         """ Add a list of additional tokens to the encoder.
#             The additional tokens are indexed starting from the last index of the
#             current vocabulary in the order of the `special_tokens` list.
#         """
#         if not special_tokens:
#             self.special_tokens = {}
#             self.special_tokens_decoder = {}
#             return
#         self.special_tokens = dict((tok, len(self.encoder) + i)
#                                    for i, tok in enumerate(special_tokens))
#         self.special_tokens_decoder = {
#             v: k
#             for k, v in self.special_tokens.items()
#         }
#         logger.info("Special tokens {}".format(self.special_tokens))
#
#     def _from_pretrained_bpe(self,
#                  vocab_file,
#                  merges_file,
#                  errors='replace',
#                  special_tokens=None,
#                  max_len=None):
#         self.max_len = max_len if max_len is not None else int(1e12)
#         self.encoder = json.load(open(vocab_file))
#         self.decoder = {v: k for k, v in self.encoder.items()}
#         self.errors = errors  # how to handle errors in decoding
#         self.byte_encoder = bytes_to_unicode()
#         self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
#         bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
#         bpe_merges = [tuple(merge.split()) for merge in bpe_data]
#         self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
#         self.cache = {}
#
#         # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
#         self.pat = re.compile(
#             r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#         )
#
#         self.special_tokens = {}
#         self.special_tokens_decoder = {}
#         self.set_special_tokens(special_tokens)
#         self.tokenizer_class = "bpe"
#
#
#     def tokenize(self, text):
#         if self.do_basic_tokenize:
#             split_tokens = []
#             for token in self.basic_tokenizer.tokenize(text):
#                 for sub_token in self.wordpiece_tokenizer.tokenize(token):
#                     split_tokens.append(sub_token)
#         else:
#             split_tokens = self.wordpiece_tokenizer.tokenize(text)
#         return split_tokens
#
#     def convert_tokens_to_ids(self, tokens):
#         """Converts a sequence of tokens into ids using the vocab."""
#         ids = []
#         for token in tokens:
#             ids.append(self.vocab[token])
#         if len(ids) > self.max_len:
#             logger.warning(
#                 "Token indices sequence length is longer than the specified maximum "
#                 " sequence length for this BERT model ({} > {}). Running this"
#                 " sequence through BERT will result in indexing errors".format(
#                     len(ids), self.max_len))
#         return ids
#
#     def convert_ids_to_tokens(self, ids):
#         """Converts a sequence of ids in wordpiece tokens using the vocab."""
#         tokens = []
#         for i in ids:
#             tokens.append(self.ids_to_tokens[i])
#         return tokens



