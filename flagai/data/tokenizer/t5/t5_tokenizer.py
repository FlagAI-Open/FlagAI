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

from transformers import T5Tokenizer
from ..tokenizer import Tokenizer, CommandToken

import jieba
import unicodedata
import re
from typing import List
"""define some default command tokens for the tokenizer to use"""


class T5BPETokenizer(Tokenizer):

    def __init__(self, tokenizer_model_type="t5-base", cache_dir=None):

        self.text_tokenizer = T5Tokenizer.from_pretrained(tokenizer_model_type,
                                                          cache_dir=cache_dir)

        self._tokens = list(self.text_tokenizer.get_vocab().keys())
        self._vocab = {k: v for k, v in self.text_tokenizer.get_vocab().items()}
        self.num_tokens = len(self._tokens)

        self._command_tokens = [
            CommandToken('unk', '[UNK]', self.get_specialid_from_text_tokenizer('unk')),
            CommandToken('eos', '[PAD]', self.get_specialid_from_text_tokenizer('pad')),
            CommandToken('sep', '[SEP]', self.num_tokens),

            CommandToken('pad', '[PAD]', self.num_tokens + 1),
            CommandToken('cls', '[CLS]', self.num_tokens + 2),
            CommandToken('MASK', '[MASK]',
                         self.num_tokens + 3),
        ]
        self._command_tokens.extend([
            CommandToken('sop', '<|startofpiece|>', self.num_tokens + 4),
            CommandToken('eop', '<|endofpiece|>', self.num_tokens + 5)
        ])
        self.num_tokens += 5

        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {
            tok.token: tok
            for tok in self._command_tokens
        }
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}

    def get_specialid_from_text_tokenizer(self, token):
        if token in ["eos", "sep"]:
            return self._vocab.get('</s>')
        elif token == "cls":
            return self._vocab.get('<|endoftext|>')
        elif token == "unk":
            return self._vocab.get('<unk>')
        elif token == "pad":
            return self._vocab.get('<pad>')
        elif token == "mask":
            return self._vocab.get('<mask>')
        else:
            raise NameError("token not exists")

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    def _encode(self, text):
        """text string to ids"""
        tokens = self.text_tokenizer._tokenize(text)
        ids = self.text_tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def EncodeAsTokens(self, text, process_fn=None):
        """convert wordpiece token to Id"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.text_tokenizer.tokenize(processed_text)
        return tokens

    def IdToToken(self, Id, type_token=False):
        """convert Id to sentencpiece token"""
        if Id in self.command_id_map:
            return self.command_id_map[Id].token
        return self.text_tokenizer.ids_to_tokens[Id]

    def TokenToId(self, token, type_token=False):
        """convert sentencpiece token to Id"""
        if isinstance(token, (CommandToken)):
            return token.Id
        return self.text_tokenizer._convert_token_to_id(token.strip())

    def DecodeIds(self, Ids):
        """converts ids to wordpiece tokens and joins them as a text string"""
        Tokens = []
        for Id in Ids:
            if Id in self.command_id_map:
                Tokens.append(self.command_id_map[Id].token)
            elif Id < self.text_tokenizer.vocab_size:
                Tokens.append(self.text_tokenizer._convert_id_to_token(Id))
        return self.text_tokenizer.convert_tokens_to_string(Tokens)
        # new_tokens = []
        # for token in Tokens:
        #     if token.startswith('##') and len(new_tokens) > 0:
        #         new_tokens[-1] += token[2:]
        #     else:
        #         new_tokens.append(token)
        # return ' '.join(new_tokens)

    def DecodeTokens(self, Tokens):
        """converts wordpiece tokens to a text string"""
        return ' '.join(Tokens)


class T5KGBPETokenizer(Tokenizer):

    def __init__(self, tokenizer_model_type="t5-base", cache_dir=None):
        """初始化
        """
        self._token_pad = '[PAD]'
        self._token_cls = '[CLS]'
        self._token_sep = '[SEP]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'

        self.text_tokenizer = T5Tokenizer.from_pretrained(tokenizer_model_type,
                                                          cache_dir=cache_dir)
        self.text_tokenizer.max_len = int(1e12)

    def _encode(self, text):
        """text string to ids"""
        tokens = self.text_tokenizer._tokenize(text)
        ids = self.text_tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def tokenize(self, text: str, add_cls=True, add_sep=True, max_length=None):
        """分词函数
        """
        tokens = self._tokenize(text)
        if add_cls:
            tokens.insert(0, self._token_cls)
        if add_sep:
            tokens.append(self._token_sep)

        if max_length is not None:
            self.truncate_sequence(max_length, tokens, None, -2)

        return tokens

    def token_to_id(self, token):
        """token转换为对应的id
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        """token序列转换为对应的id序列
        """
        return [self.token_to_id(token) for token in tokens]

    def truncate_sequence(self,
                          max_length,
                          first_sequence: List[str],
                          second_sequence=None,
                          pop_index=-1):
        """截断总长度
        """
        if second_sequence is None:
            second_sequence = []

        while True:
            total_length = len(first_sequence) + len(second_sequence)
            if total_length <= max_length:
                break
            elif len(first_sequence) > len(second_sequence):
                first_sequence.pop(pop_index)
            else:
                second_sequence.pop(pop_index)

    def encode(self,
               first_text,
               second_text=None,
               max_length=None,
               first_length=None,
               second_length=None,
               is_segment=True):
        """输出文本对应token id和segment id
        如果传入first_length，则强行padding第一个句子到指定长度；
        同理，如果传入second_length，则强行padding第二个句子到指定长度。
        """

        first_tokens = self.tokenize(first_text)

        if second_text is None:
            second_tokens = None
        else:
            second_tokens = self.tokenize(second_text, add_cls=False)

        if max_length is not None:
            self.truncate_sequence(max_length, first_tokens, second_tokens, -2)

        first_token_ids = self.tokens_to_ids(first_tokens)
        if first_length is not None:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self._token_pad_id] *
                                   (first_length - len(first_token_ids)))
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            second_token_ids = self.tokens_to_ids(second_tokens)
            if second_length is not None:
                second_token_ids = second_token_ids[:second_length]
                second_token_ids.extend(
                    [self._token_pad_id] *
                    (second_length - len(second_token_ids)))
            second_segment_ids = [1] * len(second_token_ids)

            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        if is_segment:
            return first_token_ids, first_segment_ids
        else:
            return first_token_ids

    def id_to_token(self, i):
        """id序列为对应的token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """id序列转换为对应的token序列
        """
        return [self.id_to_token(i) + " " for i in ids]

    def decode(self, ids):
        """转为可读文本
        """
        raise NotImplementedError

    def _tokenize(self, text):
        """基本分词函数
        """
        raise NotImplementedError


class T5JiebaTokenizer(T5BPETokenizer):

    def __init__(self,
                 token_dict,
                 pre_tokenizer=lambda x: jieba.cut(x, HMM=False)):
        super(T5JiebaTokenizer, self).__init__()
        self.pre_tokenizer = pre_tokenizer

        self._token_dict = token_dict

        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        for token in ['pad', 'cls', 'sep', 'unk', 'mask']:
            try:
                _token_id = token_dict[getattr(self, "_token_" + str(token))]
                # print(_token_id)
                setattr(self, "_token_" + str(token) + "_id", _token_id)
                self.token_start_id = self._token_cls_id
                self.token_end_id = self._token_sep_id
            except Exception:
                pass
        self._vocab_size = len(token_dict)

    def token_to_id(self, token):
        """token转换为对应的id
        """
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, i):
        """id转换为对应的token
        """
        return self._token_dict_inv[i]

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping

    def decode(self, ids, tokens=None):
        """转为可读文本
        """
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]

        text = ''
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token

        text = re.sub(' +', ' ', text)
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)

        return text.strip()

    # def decode(self, ids):
    #     """转为可读文本
    #     """
    #     tokens = self.ids_to_tokens(ids)

    #     return "".join(tokens).strip()

    def _tokenize(self, text):
        """基本分词函数
        """
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        spaced = ''
        for ch in text:
            # print(ch)
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))

        return tokens

    def _word_piece_tokenize(self, word):
        """word内分成subword
        """
        if word in self._token_dict:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop

        return tokens

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
               unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\xb7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')


def load_chinese_base_vocab(vocab_path,
                            simplfied=False,
                            startswith=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]):
    """
    加载官方中文bert模型字典
    simplified: 是否简化词典
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    word2idx = {}
    for index, line in enumerate(lines):
        word2idx[line.strip("\n")] = index

    if simplfied:
        new_token_dict, keep_tokens = {}, []
        for t in startswith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(word2idx[t])

        for t, _ in sorted(word2idx.items(), key=lambda s: s[1]):
            if t not in new_token_dict:
                keep = True
                if len(t) > 1:
                    for c in Tokenizer.stem(t):
                        if (Tokenizer._is_cjk_character(c)
                                or Tokenizer._is_punctuation(c)):
                            keep = False
                            break
                if keep:
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(word2idx[t])

        print("精简后的词表大小为：" + str(len(keep_tokens)))
        return new_token_dict, keep_tokens
    else:
        return word2idx
