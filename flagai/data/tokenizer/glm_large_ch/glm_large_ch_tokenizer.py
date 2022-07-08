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

import torch
from ..tokenizer import GLMTokenizer, TypeToken, CommandToken
from . import glm_large_ch

print_rank_0 = print


class GLMLargeChTokenizer(GLMTokenizer):

    def __init__(self,
                 vocab_path=None,
                 add_block_symbols=True,
                 add_task_mask=True,
                 add_decoder_mask=False,
                 fix_command_token=True):
        """
        Args:
            add_block_symbols: (str):
                When add_block_symbol is True, a bunch of block-masking-related special tokens will be added to the vocab
            add_task_mask: (bool)
                when add_task_mask is True, the generation mask token and gap sentence mask token will be distinguished
            add_decoder_mask (bool)
                When add_decoder_mask is True, some tokens of the block spans will be masked for BERT, and a special token
                    for that will be added to vocab
            fix_command_token: (bool)
                When add_task_mask, setting fix_command
        """
        self.text_tokenizer = glm_large_ch.from_pretrained(vocab_path)
        self.num_command_tokens = 0
        self.num_text_tokens = self.text_tokenizer.sp.vocab_size()
        self.num_tokens = self.num_text_tokens
        self.num_type_tokens = 2

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
        if torch.cuda.is_available():
            print_rank_0({tok.name: tok.Id for tok in self._command_tokens})
        self.type_tokens = [
            TypeToken('str0', '<str0>', 0),
            TypeToken('str1', '<str1>', 1),
        ]
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}

        # self._tokens = list(self.text_tokenizer.encoder.keys())
        # self._vocab = {k:v for k,v in self.text_tokenizer.encoder.items()}
        #
        # self._text_tokens = list(self._tokens)
        # self._text_token_vocab = {k:v for k,v in self.text_tokenizer.encoder.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {
            t: Id
            for Id, t in self.command_id_map.items()
        }

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def _encode(self, text):
        ids = self.text_tokenizer.encode(text)
        return ids

    def encode_plus(  #for Seq2seq
            self,
            source_text,
            target_text=None,
    ):

        sop_id = self.get_command_id('sop')  #start of piece
        eop_id = self.get_command_id('eop')  #end of piece
        sep_id = self.get_command_id('sep')  #seperation

        source_tokens = self.EncodeAsIds(source_text)
        source_tokens = [sop_id] + source_tokens + [sep_id]

        # no pading for consistency
        len_source = len(source_tokens)
        sop_pos = source_tokens.index(sop_id)
        loss_mask = [0] * len_source
        block_position_ids = [0] * len_source
        position_ids = list(range(len_source))

        if target_text:
            target_tokens = self.EncodeAsIds(target_text)
            target_tokens = target_tokens + [eop_id]
            loss_mask += [1] * len(target_tokens)
            block_position_ids += [0] * len(target_tokens)
            position_ids += [x + len_source for x in range(len(target_tokens))]
            tokens = source_tokens + target_tokens
            position_ids = [position_ids[:-1], block_position_ids[:-1]]
            sample = {
                'input_ids': tokens[:-1],
                'target_ids': tokens[1:],
                'attention_mask': sop_pos,
                'loss_mask': loss_mask[:-1],
                "position_ids": position_ids
            }
        else:
            position_ids = [position_ids, block_position_ids]
            sample = {
                'input_ids': source_tokens,
                'attention_mask': sop_pos,
                "position_ids": position_ids,
                'loss_mask': loss_mask,
            }
        return sample

    def MultiWordId(self, exception=None):
        #get multi word tokens' ids
        #return ids list
        #exception token: string list
        result = []
        for i in range(self.num_text_tokens):
            word = self.IdToToken(i)
            if exception:
                if word not in exception and len(word) > 2:
                    result.append(i)
            else:
                if len(word) > 2:
                    result.append(i)
        return result

    def CommandTokenIds(self, exception=None):
        result = []
        for s in self._command_tokens:
            if not exception or (exception and s.name not in exception):
                result.append(s.Id)
        return (result)

    def EncodeAsTokens(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.text_tokenizer.tokenize(processed_text)
        return tokens

    def IdToToken(self, Id, type_token=False):
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id in self.command_id_map:
            return self.command_id_map[Id].token
        elif Id in self.type_id_map:
            return self.type_id_map[Id].token
        else:
            return self.text_tokenizer.convert_id_to_token(int(Id))

    def TokenToId(self, token, type_token=False):
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        return self.text_tokenizer.convert_token_to_id(token)

    def DecodeIds(self, Ids, type_token=False):
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.
                            type_id_map[Id].token for Id in Ids)
        Ids = list(map(int, Ids))
        pieces = []
        last = 0
        for i, token_id in enumerate(Ids):
            if token_id in self.command_id_map:
                pieces.append(Ids[last:i])
                pieces.append(token_id)
                last = i + 1
        pieces.append(Ids[last:])
        text = ""
        for piece in pieces:
            if isinstance(piece, int):
                text += self.command_id_map[piece].token
            elif piece:
                text += self.text_tokenizer.decode(piece)
        return text

    def DecodeTokens(self, Tokens, type_token=False):
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t
                            for t in Tokens)
        return self.text_tokenizer.decode(
            [self.TokenToId(tok) for tok in Tokens])
