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

"""This code is adapted from OpenAI Megatron for CPM2 tokenizer"""
# TODO will change to our SAM implementation in the future

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from io import open
import jieba
import collections
import six
import os
from typing import Union

from model_center.utils import check_web_and_convert_path

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with open(vocab_file, "r", encoding="utf-8") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab

def is_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def is_contain_point(check_str):
    for ch in check_str:
        if u'0' <= ch <= u'9':
            return True
    return False


class WordpieceTokenizer(object):

    def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token):

        token = convert_to_unicode(token)

        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]

        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if is_contain_chinese(substr) or is_contain_point(substr):
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                else:
                    # if start > 0:
                    #     substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                end -= 1
            if cur_substr is None:
                sub_tokens.append(self.unk_token)
                start += 1
                continue
            sub_tokens.append(cur_substr)
            start = end

        return sub_tokens


class CPM2Tokenizer(object): # TODO

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike]):
        path = check_web_and_convert_path(pretrained_model_name_or_path, 'tokenizer')
        return cls(os.path.join(path, 'vocab.txt'), space_token = "▂", line_token = "▃")

    def __init__(self, 
                 vocab_file, 
                 max_sentinels = 190, 
                 max_len = None, 
                 q2b = False,
                 sod_token = '<s>',
                 eod_token = '<eod>',
                 pad_token = '<pad>',
                 unk_token = '<unk>',
                 line_token = '</n>',
                 space_token = '</_>'):

        self.sod_token = sod_token
        self.eod_token = eod_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.line_token = line_token
        self.space_token = space_token

        self.max_len = max_len if max_len is not None else int(1e12)

        self.encoder = load_vocab(vocab_file)
        self.encoder[self._space_token] = len(self.encoder)
        self.encoder[self._line_token] = len(self.encoder)
        self.decoder = {v : k for k, v in self.encoder.items()}

        self.wordpiece_tokenizer = WordpieceTokenizer(vocab = self.encoder, 
                                                      unk_token = self.unk_token)

        self.trans_common = str.maketrans(" \n", "\u2582\u2583")
        BAN1='｡､｢｣'
        QUAN1='。、「」'
        # 全角转半角
        QUAN2='\u3000＂＇＾｀，：；？！「」『』（）｛｝［］〔〕＜＞〜｜＼＿＄％＃＆＠＝＊／＋－．０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        BAN2 =' "\'^`,:;?!“”‘’(){}[][]<>~|\_$%#&@=*/+-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        # 暂不转换的全半角
        # FULL = " ゛´—‐‘’“”〈〉′″"
        # half='''
        #  "'--''""[]
        # '''
        self.trans_qb_1 = str.maketrans(QUAN1, BAN1)
        self.trans_qb_2 = str.maketrans(QUAN2, BAN2)
        self.q2b = q2b

        self.sentinel_list = [
            self.encoder['<s_{}>'.format(i)] 
                for i in range(max_sentinels)]

        self.en_vocab = {}
        for k, v in self.encoder.items():
            if is_contain_chinese(k):
                self.en_vocab[v] = False
            else:
                self.en_vocab[v] = True

        self._space_id = self.encoder[self._space_token]
        self.en_vocab[self._space_id] = False

        self._line_id = self.encoder[self._line_token]
        self.en_vocab[self._line_id] = False

        self.space_id = self.encoder[self.space_token]
        self.line_id = self.encoder[self.line_token]

    @property
    def vocab_size(self):
        return len(self.encoder) - 2

    @property
    def sod_id(self):
        return self.encoder[self.sod_token]

    @property
    def eod_id(self):
        return self.encoder[self.eod_token]

    @property
    def pad_id(self):
        return self.encoder[self.pad_token]

    @property
    def unk_id(self):
        return self.encoder[self.unk_token]

    @property
    def _space_token(self):
        return '\uFFF2'

    @property
    def _line_token(self):
        return '\uFFF3'

    def __len__(self):
        return len(self.encoder)

    def get_sentinel_num(self):
        return len(self.sentinel_list)

    def get_sentinel_id(self, idx):
        return self.sentinel_list[idx]

    def _translate(self, sent):
        if self.q2b:
            return sent.translate(self.trans_common) \
                       .translate(self.trans_qb_1) \
                       .translate(self.trans_qb_2)
        else:
            return sent.translate(self.trans_common)

    def tokenize(self, text):
        """ Tokenize a string. """
        output_tokens = []
        for x in jieba.cut(text, cut_all=False):
            output_tokens.extend(
                self.wordpiece_tokenizer.tokenize(
                    self._translate(x)
                )
            )
        return output_tokens

    def encode(self, text):
        """ Encode a string into ids. """
        text = text.replace("\n", self._line_token) \
                   .replace(self.line_token, self._line_token) \
                   .replace(" ", self._space_token) \
                   .replace(self.space_token, self._space_token)
        new_output_tokens = []
        for x in self.tokenize(text):
            x = self.encoder[x]
            if x == self._space_id:
                new_output_tokens.append(self.space_id)
            elif x == self._line_id:
                new_output_tokens.append(self.line_id)
            else:
                new_output_tokens.append(x)
        return new_output_tokens

    def decode(self, tokens):
        """ Decode ids into a string. """
        text = ''.join([self.decoder[x] for x in tokens])
        text = text.replace(self._space_token, ' ') \
                   .replace(self._line_token, '\n') \
                   .replace(self.space_token, ' ') \
                   .replace(self.line_token, '\n')
        return text

    def check(self, token):
        return token in self.encoder

    def convert_tokens_to_ids(self, tokens):
        return [self.encoder.get(x, self.encoder[self.unk_token]) for x in tokens]

    def convert_ids_to_tokens(self, ids):
        _ids = ids.cpu().numpy()
        return [self.decoder[x] for x in _ids]
