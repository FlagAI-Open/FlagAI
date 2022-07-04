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


import itertools
import logging
logger = logging.getLogger(__name__)
import os
from flagai.data.tokenizer.glm_large_en.wordpiece import load_vocab, BasicTokenizer, whitespace_tokenize
import collections
from flagai.data.tokenizer.tokenizer import TypeToken, CommandToken
from wp_tokenizer import WordpieceTokenizer
from bpe_tokenizer import BPETokenizer
from base_tokenizer import BaseTokenizer
import torch


class Tokenizer(BaseTokenizer):
    def __init__(self,
                 tokenizer_model_type='GLM-large-en',
                 cache_dir=None,
                 **kwargs):
        super().__init__(**kwargs)
        if self.tokenizer_class == "wp":
            self.text_tokenizer = WordpieceTokenizer(self.vocab_file)
        elif self.tokenizer_class == "bpe":
            self.text_tokenizer = BPETokenizer(self.vocab_file, self.merges_file)
        elif self.tokenizer_class == "sp":
            pass
        self.text_tokenizer = WordpieceTokenizer(self.vocab_file, self.merges_file)  # temporary
        # default to bert-large-uncased tokenizer

        if not torch.distributed.is_initialized(
        ) or torch.distributed.get_rank() == 0:
            print('loading GLMBertWordPieceTokenizer (', tokenizer_model_type,
                  ') from cache_dir ', cache_dir)

        if not torch.distributed.is_initialized(
        ) or torch.distributed.get_rank() == 0:
            print('loaded', tokenizer_model_type)

    def EncodeAsIds(self, text: str, process_fn=None):
        """Input text string => a list of token ids"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)

        tokens = self.EncodeAsTokens(processed_text, process_fn=process_fn)
        Ids = [self.TokenToId(token) for token in tokens]
        return Ids

    def EncodeAsTokens(self, text: str, process_fn=None):
        """Input text string => a list of tokens"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.text_tokenizer.tokenize(processed_text)
        return tokens

    def IdToToken(self, Id: int):
        """Token id => token"""
        return self.text_tokenizer.ids_to_tokens[Id]

    def TokenToId(self, token: str):
        """Token => token id"""
        try:
            return self.text_tokenizer.vocab[token]
        except KeyError:
            return self.text_tokenizer.vocab[token.strip()]

    def DecodeIds(self, Ids):
        """A list of token ids => recovered text string"""
        return self.DecodeTokens([self.IdToToken(id) for id in Ids])

    def DecodeTokens(self, tokens):
        """A list of tokens => recovered text string"""
        return ' '.join(tokens)




class GLMTokenizer(Tokenizer):
    def __init__(self,
                 tokenizer_model_type='GLM-large-en',
                 cache_dir=None,
                 add_block_symbols=True,
                 add_sentinel_token=0,
                 add_task_mask=True,
                 add_decoder_mask=False,
                 **kwargs):
        super().__init__(**kwargs)

        # set command tokens from wordpiece tokenizer values
        self.num_command_tokens = 6
        self.num_tokens = len(self.text_tokenizer.vocab)
        self.num_text_tokens = self.num_tokens - 5
        self.num_type_tokens = 2

        self._command_tokens = [
            CommandToken('pad', '[PAD]', self.text_tokenizer.vocab['[PAD]']),
            CommandToken('ENC', '[CLS]', self.text_tokenizer.vocab['[CLS]']),
            CommandToken('MASK', '[MASK]',
                         self.text_tokenizer.vocab['[MASK]']),
            CommandToken('unk', '[UNK]', self.text_tokenizer.vocab['[UNK]']),
            CommandToken('sep', '[SEP]', self.text_tokenizer.vocab['[SEP]']),
            CommandToken('eos', '[PAD]', self.text_tokenizer.vocab['[PAD]']),
        ]
        if add_block_symbols:
            self._command_tokens.extend([
                CommandToken('sop', '<|startofpiece|>', self.num_tokens),
                CommandToken('eop', '<|endofpiece|>', self.num_tokens + 1)
            ])
            self.num_tokens += 2
            self.num_command_tokens += 2
            if add_task_mask:
                self._command_tokens.extend([
                    CommandToken('gMASK', '[gMASK]', self.num_tokens),
                    CommandToken('sMASK', '[sMASK]', self.num_tokens + 1)
                ])
                self.num_tokens += 2
                self.num_command_tokens += 2
            if add_decoder_mask:
                self._command_tokens.extend(
                    [CommandToken('dBLOCK', '[dBLOCK]', self.num_tokens)])
                self.num_tokens += 1
                self.num_command_tokens += 1
        if add_sentinel_token > 0:
            for i in range(1, add_sentinel_token):
                self._command_tokens.extend([
                    CommandToken(f'MASK{i}', f'[MASK{i}]', self.num_tokens),
                    CommandToken(f'sop{i}', f'<|startofpiece{i}|>',
                                 self.num_tokens + 1)
                ])
                self.num_tokens += 2
                self.num_command_tokens += 2
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {
            tok.token: tok
            for tok in self._command_tokens
        }
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}

        # set type tokens
        self.type_tokens = [
            TypeToken('str0', '<str0>', 0),
            TypeToken('str1', '<str1>', 1),
        ]
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}

        # parse tokens and vocabs from tokenizer

        self._tokens = list(self.text_tokenizer.vocab.keys())
        self._vocab = {k: v for k, v in self.text_tokenizer.vocab.items()}

        self._text_tokens = list(self._tokens)
        self._text_token_vocab = {
            k: v
            for k, v in self.text_tokenizer.vocab.items()
        }

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {
            t: Id
            for Id, t in self.command_id_map.items()
        }

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def _encode(self, text):
        tokens = self.text_tokenizer.tokenize(text)
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
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id in self.command_id_map:
            return self.command_id_map[Id].token
        return self.text_tokenizer.ids_to_tokens[Id]

    def TokenToId(self, token, type_token=False):
        """convert sentencpiece token to Id"""
        token = token.lower()
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        try:
            return self.text_tokenizer.vocab[token]
        except KeyError:
            return self.text_tokenizer.vocab[token.strip()]

    def DecodeIds(self, Ids, type_token=False):
        """converts ids to wordpiece tokens and joins them as a text string"""
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.
                            type_id_map[Id].token for Id in Ids)
        Tokens = []
        for Id in Ids:
            if Id in self.command_id_map:
                Tokens.append(self.command_id_map[Id].token)
            elif Id in self.text_tokenizer.ids_to_tokens:
                Tokens.append(self.text_tokenizer.ids_to_tokens[Id])
        new_tokens = []
        for token in Tokens:
            if token.startswith('##') and len(new_tokens) > 0:
                new_tokens[-1] += token[2:]
            else:
                new_tokens.append(token)
        return ' '.join(new_tokens)

    def DecodeTokens(self, Tokens, type_token=False):
        """converts wordpiece tokens to a text string"""
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t
                            for t in Tokens)
        return ' '.join(Tokens)

    def EncodeAsIds(self, text, process_fn=None):
        """
        encode text using text tokenizer and shift Id values for command tokens
        """
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)

        def split_on_token(tok_extended: CommandToken, text):
            result = []
            tok = tok_extended.token
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # CommandToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # Strip white spaces on the right
                if tok_extended.rstrip and i > 0:
                    # A bit counter-intuitive but we strip the left of the string
                    # since tok_extended.rstrip means the special token is eating all white spaces on its right
                    sub_text = sub_text.lstrip()
                # Strip white spaces on the left
                if tok_extended.lstrip and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()  # Opposite here

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self.text_tokenizer.encode(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self._command_token_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (self._encode(token)
                     if token not in self._command_token_tokens else
                     [self.command_token_map[token].Id]
                     for token in tokenized_text)))

        no_split_tokens = self._command_tokens
        Ids = split_on_tokens(no_split_tokens, processed_text)
        return Ids

if __name__ == '__main__':
    tokenizer = GLMTokenizer.from_pretrained('GLM-large-en')
    # tokenizer = Tokenizer.from_pretrained('GLM-large-en')
    print(tokenizer.EncodeAsIds("fried chicken makes me happy"))



