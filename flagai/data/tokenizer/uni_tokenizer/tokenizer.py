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
from flagai.data.tokenizer.tokenizer import CommandToken
from flagai.data.tokenizer.uni_tokenizer.wp_tokenizer import WordpieceTokenizer
from flagai.data.tokenizer.uni_tokenizer.bpe_tokenizer import BPETokenizer
from flagai.data.tokenizer.uni_tokenizer.sp_tokenizer import SentencePieceTokenizer
from flagai.data.tokenizer.uni_tokenizer.base_tokenizer import BaseTokenizer
# import torch


# class Tokenizer(BaseTokenizer):
#     def __init__(self,
#                  **kwargs):
#         super().__init__(**kwargs)
#
#         if self.tokenizer_class == "wp":
#             self.text_tokenizer = WordpieceTokenizer(self.vocab_file)
#         elif self.tokenizer_class == "bpe":
#             self.text_tokenizer = BPETokenizer(self.vocab_file, self.merges_file)
#         elif self.tokenizer_class == "sp":
#             self.text_tokenizer = SentencePieceTokenizer(self.sp_model_file)
#
#         self.num_tokens = self.text_tokenizer.vocab_size
#         self.command_name_map = {}
#
#         if not torch.distributed.is_initialized(
#         ) or torch.distributed.get_rank() == 0:
#             print('loading GLMBertWordPieceTokenizer (', self.tokenizer_model_name,
#                   ') from cache_dir ', self.cache_dir)
#             print('loaded', self.tokenizer_model_name)
#
#     def __len__(self):
#         """total number of tokens"""
#         return self.num_tokens
#
#     def get_command_id(self, name):
#         """get command token corresponding to `name`"""
#         return self.command_name_map[name].Id
#
#     def EncodeAsIds(self, text: str):
#         """Input text string => a list of token ids"""
#         tokens = self.EncodeAsTokens(text)
#         ids = self.text_tokenizer.convert_tokens_to_ids(tokens)
#         return ids
#
#     def EncodeAsTokens(self, text: str):
#         """Input text string => a list of tokens"""
#         tokens = self.text_tokenizer.tokenize(text)
#         return tokens
#
#     def IdToToken(self, id: int):
#         """Token id => token"""
#         return self.text_tokenizer.convert_ids_to_tokens([id])[0]
#
#     def TokenToId(self, token: str):
#         """Token => token id"""
#         try:
#             return self.text_tokenizer.convert_tokens_to_ids(token)[0]
#         except KeyError:
#             return self.text_tokenizer.convert_tokens_to_ids(token.strip())[0]
#
#     def DecodeIds(self, ids):
#         """A list of token ids => recovered text string"""
#         return self.DecodeTokens([self.text_tokenizer.convert_ids_to_tokens(ids)])
#
#     def DecodeTokens(self, tokens):
#         """A list of tokens => recovered text string"""
#         return self.text_tokenizer.convert_tokens_to_string(tokens)


class Tokenizer(BaseTokenizer):
    def __init__(self,
                 add_block_symbols=True,
                 add_sentinel_token=0,
                 add_task_mask=True,
                 add_decoder_mask=False,
                 fix_command_token=True,
                 **kwargs):
        super().__init__(**kwargs)

        if self.tokenizer_class == "wp":
            self.text_tokenizer = WordpieceTokenizer(self.vocab_file)
        elif self.tokenizer_class == "bpe":
            self.text_tokenizer = BPETokenizer(self.vocab_file, self.merges_file)
        elif self.tokenizer_class == "sp":
            self.text_tokenizer = SentencePieceTokenizer(self.sp_model_file)
        else:
            raise NotImplementedError("cannot assign a tokenize class")
        self.num_tokens = self.text_tokenizer.vocab_size

        if self.tokenizer_class == "wp":
            # set command tokens from wordpiece tokenizer values
            self.num_command_tokens = 6
            # self.num_tokens = len(self.text_tokenizer.vocab)
            self.num_text_tokens = self.num_tokens - 5
            self.num_type_tokens = 2

            self._command_tokens = [
                CommandToken('pad', '[PAD]', self.text_tokenizer.convert_token_to_id('[PAD]')),
                CommandToken('cls', '[CLS]', self.text_tokenizer.convert_token_to_id('[CLS]')),
                CommandToken('MASK', '[MASK]',
                             self.text_tokenizer.convert_token_to_id('[MASK]')),
                CommandToken('unk', '[UNK]', self.text_tokenizer.convert_token_to_id('[UNK]')),
                CommandToken('sep', '[SEP]', self.text_tokenizer.convert_token_to_id('[SEP]')),
                CommandToken('eos', '[PAD]', self.text_tokenizer.convert_token_to_id('[PAD]')),
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
        elif self.tokenizer_class == "bpe":
            if self.tokenizer_model_name.startswith('roberta'):
                self.num_command_tokens = 6
                self.num_text_tokens = self.num_tokens - 3
                self._command_tokens = [
                    CommandToken('pad', '<|endoftext|>',
                                 self.text_tokenizer.convert_token_to_id('</s>')),
                    CommandToken('eos', '<|endoftext|>',
                                 self.text_tokenizer.convert_token_to_id('</s>')),
                    CommandToken('sep', '[SEP]',
                                 self.text_tokenizer.convert_token_to_id('</s>')),
                    CommandToken('cls', '[CLS]',
                                 self.text_tokenizer.convert_token_to_id('<s>')),
                    CommandToken('MASK',
                                 '[MASK]',
                                 self.text_tokenizer.convert_token_to_id('<mask>'),
                                 lstrip=True),
                    CommandToken('unk', '[UNK]',
                                 self.text_tokenizer.convert_token_to_id('<unk>'))
                ]
                if add_block_symbols:
                    self._command_tokens.extend([
                        CommandToken('sop', '<|startofpiece|>', self.num_tokens),
                        CommandToken('eop', '<|endofpiece|>', self.num_tokens + 1)
                    ])
                    self.num_tokens += 2
                    self.num_command_tokens += 2
            else:
                self.num_command_tokens = 2
                self.num_text_tokens = self.num_tokens - 1
                self._command_tokens = [
                    CommandToken('pad', '<|endoftext|>',
                                 self.text_tokenizer.convert_token_to_id('<|endoftext|>')),
                    CommandToken('eos', '<|endoftext|>',
                                 self.text_tokenizer.convert_token_to_id('<|endoftext|>'))
                ]
                if add_block_symbols:
                    self._command_tokens.extend([
                        CommandToken('sop', '<|startofpiece|>', self.num_tokens),
                        CommandToken('eop', '<|endofpiece|>', self.num_tokens + 1),
                        CommandToken('cls', '[CLS]', self.num_tokens + 2),
                        CommandToken('MASK',
                                     '[MASK]',
                                     self.num_tokens + 3,
                                     lstrip=True),
                        CommandToken('sep', '[SEP]', self.num_tokens + 4),
                        CommandToken('unk', '[UNK]', self.num_tokens + 5)
                    ])
                    self.num_tokens += 6
                    self.num_command_tokens += 6
            if add_block_symbols:
                if add_task_mask:
                    self._command_tokens.extend([
                        CommandToken('gMASK',
                                     '[gMASK]',
                                     self.num_tokens,
                                     lstrip=True),
                        CommandToken('sMASK',
                                     '[sMASK]',
                                     self.num_tokens + 1,
                                     lstrip=True)
                    ])
                    self.num_tokens += 2
                    self.num_command_tokens += 2
                if add_decoder_mask:
                    self._command_tokens.extend(
                        [CommandToken('dBLOCK', '[dBLOCK]', self.num_tokens)])
                    self.num_tokens += 1
                    self.num_command_tokens += 1
        elif self.tokenizer_class == "sp":
            self.num_command_tokens = 0
            self.num_text_tokens = self.text_tokenizer.vocab_size
            self.num_tokens = self.num_text_tokens

            self._command_tokens = [
                CommandToken('pad', '<|endoftext|>', self.num_text_tokens),
                CommandToken('eos', '<|endoftext|>', self.num_text_tokens),
                CommandToken('sep', '[SEP]', self.num_text_tokens + 1),
                CommandToken('cls', '[CLS]', self.num_text_tokens + 2),
                CommandToken('MASK',
                             '[MASK]',
                             self.num_text_tokens + 3,
                             lstrip=True),
                CommandToken('unk', '[UNK]', self.num_text_tokens + 4)
            ]
            self.num_tokens += 5
            self.num_command_tokens += 6
            if add_block_symbols:
                self._command_tokens.extend([
                    CommandToken('sop', '<|startofpiece|>', self.num_tokens + 1),
                    CommandToken('eop', '<|endofpiece|>', self.num_tokens + 2)
                ])
                if fix_command_token:
                    self.num_tokens += 3
                else:
                    self.num_tokens += 2
                self.num_command_tokens += 2
                if add_task_mask:
                    if fix_command_token:
                        self._command_tokens.extend([
                            CommandToken('sMASK',
                                         '[sMASK]',
                                         self.num_tokens,
                                         lstrip=True),
                            CommandToken('gMASK',
                                         '[gMASK]',
                                         self.num_tokens + 1,
                                         lstrip=True)
                        ])
                    else:
                        self._command_tokens.extend([
                            CommandToken('gMASK',
                                         '[gMASK]',
                                         self.num_tokens,
                                         lstrip=True),
                            CommandToken('sMASK',
                                         '[sMASK]',
                                         self.num_tokens + 1,
                                         lstrip=True)
                        ])
                    self.num_tokens += 2
                    self.num_command_tokens += 2
                if add_decoder_mask:
                    self._command_tokens.extend(
                        [CommandToken('dBLOCK', '[dBLOCK]', self.num_tokens)])
                    self.num_tokens += 1
                    self.num_command_tokens += 1
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {
            tok.token: tok
            for tok in self._command_tokens
        }
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}
        self._command_token_tokens = list(self.command_token_map.keys())

    def get_command_id(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name].Id

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

    def IdToToken(self, id):
        """convert Id to sentencpiece token"""
        if isinstance(id, (CommandToken)):
            return id.token
        if id in self.command_id_map:
            return self.command_id_map[id].token
        return self.text_tokenizer.convert_id_to_token(id)

    def TokenToId(self, token):
        """convert sentencpiece token to Id"""
        token = token.lower()
        if isinstance(token, (CommandToken)):
            return token.Id
        try:
            return self.text_tokenizer.convert_token_to_id(token)
        except KeyError:
            return self.text_tokenizer.convert_token_to_id(token.strip())

    def DecodeIds(self, ids):
        """converts ids to wordpiece tokens and joins them as a text string"""
        tokens = []
        for id in ids:
            if id in self.command_id_map:
                tokens.append(self.command_id_map[id].token)
            else:
                try:
                    tokens.extend(self.text_tokenizer.convert_ids_to_tokens([id]))
                except KeyError:
                    pass
        return self.text_tokenizer.convert_tokens_to_string(tokens, self.command_token_map)


    def DecodeTokens(self, tokens):
        """converts wordpiece tokens to a text string"""
        return self.text_tokenizer.convert_tokens_to_string(tokens, self.command_token_map)

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
    tokenizer = Tokenizer.from_pretrained('GLM-large-ch')
    # tokenizer = GLMTokenizer.from_pretrained('GLM-large-en')
    # print(tokenizer.EncodeAsIds("fried chicken makes me happy"))
    # print(tokenizer.EncodeAsTokens("fried chicken makes me happy deglobalization"))
    print(tokenizer.EncodeAsIds("今天吃饭吃了肯德基"))

    tokenizer = GLMTokenizer.from_pretrained('GLM-large-en')
    # print(tokenizer.EncodeAsTokens("fried chicken makes me happy"))
    # print(tokenizer.EncodeAsIds("fried chicken makes me happy"))
    print(len(tokenizer))

