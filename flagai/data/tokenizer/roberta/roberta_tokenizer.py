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

from transformers import RobertaTokenizer
from ..tokenizer import Tokenizer, CommandToken, TypeToken
import os
"""define some default command tokens for the tokenizer to use"""


class ROBERTATokenizer(Tokenizer):

    def __init__(self, tokenizer_model_type="roberta-base", cache_dir=None):
        self.text_tokenizer = RobertaTokenizer.from_pretrained(
            tokenizer_model_type, cache_dir=cache_dir)
        self.text_tokenizer.max_len = int(1e12)

        # # parse tokens and vocabs from tokenizer
        self._tokens = list(self.text_tokenizer.get_vocab().keys())
        self._vocab = {k: v for k, v in self.text_tokenizer.get_vocab().items()}
        self.num_tokens = len(self._tokens)

        self._command_tokens = [
            CommandToken('pad', '[PAD]', self.get_specialid_from_text_tokenizer('pad')),
            CommandToken('cls', '[CLS]', self.get_specialid_from_text_tokenizer('cls')),
            CommandToken('MASK', '[MASK]',
                         self.get_specialid_from_text_tokenizer('mask')),
            CommandToken('unk', '[UNK]', self.get_specialid_from_text_tokenizer('unk')),
            CommandToken('sep', '[SEP]', self.get_specialid_from_text_tokenizer('sep')),
            CommandToken('eos', '[PAD]', self.get_specialid_from_text_tokenizer('pad')),
        ]
        self._command_tokens.extend([
            CommandToken('sop', '<|startofpiece|>', self.num_tokens),
            CommandToken('eop', '<|endofpiece|>', self.num_tokens + 1)
        ])
        self.num_tokens += 2

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
            return self._vocab.get('<s>')
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
        try:
            return self._vocab[token]
        except KeyError:
            return self._vocab[token.strip()]

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
