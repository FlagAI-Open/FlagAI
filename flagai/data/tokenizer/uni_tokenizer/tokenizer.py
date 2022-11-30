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
import torch 
logger = logging.getLogger(__name__)
from flagai.data.tokenizer.tokenizer import CommandToken
from flagai.data.tokenizer.uni_tokenizer.wp_tokenizer import WordpieceTokenizer
from flagai.data.tokenizer.uni_tokenizer.bpe_tokenizer import BPETokenizer, MMBPETokenizer
from flagai.data.tokenizer.uni_tokenizer.sp_tokenizer import SentencePieceTokenizer
from flagai.data.tokenizer.uni_tokenizer.base_tokenizer import BaseTokenizer
from flagai.data.tokenizer.uni_tokenizer.difffusion_bert_tokenizer import FullTokenizer
from typing import List, Union, Optional
import unicodedata


def is_control(ch):
    """
    https://en.wikipedia.org/wiki/Control_character
    https://www.fileformat.info/info/unicode/category/Cc/index.htm
    https://www.fileformat.info/info/unicode/category/Cf/index.htm

    """
    return unicodedata.category(ch) in ('Cc', 'Cf')


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
            if self.tokenizer_model_name.lower().endswith("ch"):
                self.text_tokenizer = WordpieceTokenizer(self.vocab_file,
                                                         is_ch=True)
            elif self.tokenizer_model_name.lower().startswith('clip-cn'):
                self.text_tokenizer = FullTokenizer(self.vocab_file)
            else:
                self.text_tokenizer = WordpieceTokenizer(self.vocab_file)
        elif self.tokenizer_class == "bpe":
            if self.tokenizer_model_name.lower().startswith('clip'):
                self.text_tokenizer = MMBPETokenizer(self.vocab_file,
                                                     self.merges_file)
            else:
                self.text_tokenizer = BPETokenizer(self.vocab_file,
                                                   self.merges_file)
        elif self.tokenizer_class == "sp":
            self.text_tokenizer = SentencePieceTokenizer(self.sp_model_file)
        else:
            raise NotImplementedError("cannot assign a tokenize class")

        self.is_glm = self.tokenizer_model_name.lower().startswith('glm')
        # self.is_clip = self.tokenizer_model_name.startswith('clip')
        self.num_tokens = self.text_tokenizer.vocab_size

        if self.tokenizer_class == "wp":
            # set command tokens from wordpiece tokenizer values
            self.num_command_tokens = 6
            self.num_text_tokens = self.num_tokens - 5
            self.num_type_tokens = 2
            self.token_start_id = None
            self.token_end_id = None
            self.token_pad_id = None
            try:
                self._command_tokens = [
                    CommandToken(
                        'pad', '[PAD]',
                        self.text_tokenizer.convert_token_to_id('[PAD]')),
                    CommandToken(
                        'cls', '[CLS]',
                        self.text_tokenizer.convert_token_to_id('[CLS]')),
                    CommandToken(
                        'MASK', '[MASK]',
                        self.text_tokenizer.convert_token_to_id('[MASK]')),
                    CommandToken(
                        'unk', '[UNK]',
                        self.text_tokenizer.convert_token_to_id('[UNK]')),
                    CommandToken(
                        'sep', '[SEP]',
                        self.text_tokenizer.convert_token_to_id('[SEP]')),
                    CommandToken(
                        'eos', '[PAD]',
                        self.text_tokenizer.convert_token_to_id('[PAD]')),
                ]
                self.token_start_id = self.text_tokenizer.convert_token_to_id(
                    '[CLS]')
                self.token_end_id = self.text_tokenizer.convert_token_to_id(
                    '[SEP]')
                self.token_pad_id = self.text_tokenizer.convert_token_to_id(
                    '[PAD]')
                self.text_tokenizer._token_cls = "[CLS]"
                self.text_tokenizer._token_sep = "[SEP]"

            except KeyError:
                self._command_tokens = [
                    CommandToken(
                        'pad', '[PAD]',
                        self.text_tokenizer.convert_token_to_id('<pad>')),
                    CommandToken(
                        'cls', '[CLS]',
                        self.text_tokenizer.convert_token_to_id('<s>')),
                    CommandToken(
                        'MASK', '[MASK]',
                        self.text_tokenizer.convert_token_to_id('<mask>')),
                    CommandToken(
                        'unk', '[UNK]',
                        self.text_tokenizer.convert_token_to_id('<unk>')),
                    CommandToken(
                        'sep', '[SEP]',
                        self.text_tokenizer.convert_token_to_id('<sep>')),
                    CommandToken(
                        'eos', '[PAD]',
                        self.text_tokenizer.convert_token_to_id('</s>')),
                ]
                self.token_start_id = self.text_tokenizer.convert_token_to_id(
                    '<s>')
                self.token_end_id = self.text_tokenizer.convert_token_to_id(
                    '</s>')
                self.token_pad_id = self.text_tokenizer.convert_token_to_id(
                    '<pad>')
                self.text_tokenizer._token_cls = "<s>"
                self.text_tokenizer._token_sep = "</s>"
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
                        CommandToken(f'MASK{i}', f'[MASK{i}]',
                                     self.num_tokens),
                        CommandToken(f'sop{i}', f'<|startofpiece{i}|>',
                                     self.num_tokens + 1)
                    ])
                    self.num_tokens += 2
                    self.num_command_tokens += 2
        elif self.tokenizer_class == "bpe":
            if self.tokenizer_model_name.lower().startswith('roberta'):
                self.num_command_tokens = 6
                self.num_text_tokens = self.num_tokens - 3
                self._command_tokens = [
                    CommandToken(
                        'pad', '<|endoftext|>',
                        self.text_tokenizer.convert_token_to_id('</s>')),
                    CommandToken(
                        'eos', '<|endoftext|>',
                        self.text_tokenizer.convert_token_to_id('</s>')),
                    CommandToken(
                        'sep', '[SEP]',
                        self.text_tokenizer.convert_token_to_id('</s>')),
                    CommandToken(
                        'cls', '[CLS]',
                        self.text_tokenizer.convert_token_to_id('<s>')),
                    CommandToken(
                        'MASK',
                        '[MASK]',
                        self.text_tokenizer.convert_token_to_id('<mask>'),
                        lstrip=True),
                    CommandToken(
                        'unk', '[UNK]',
                        self.text_tokenizer.convert_token_to_id('<unk>'))
                ]
                if add_block_symbols:
                    self._command_tokens.extend([
                        CommandToken('sop', '<|startofpiece|>',
                                     self.num_tokens),
                        CommandToken('eop', '<|endofpiece|>',
                                     self.num_tokens + 1)
                    ])
                    self.num_tokens += 2
                    self.num_command_tokens += 2
                self.token_end_id = self.text_tokenizer.convert_token_to_id(
                    '</s>')
            elif self.tokenizer_model_name.lower().startswith('clip'):
                self.num_command_tokens = 2
                self._command_tokens = [
                    CommandToken(
                        'sot', '<start_of_text>',
                        self.text_tokenizer.convert_token_to_id('</s>')),
                    CommandToken(
                        'eot', '<end_of_text>',
                        self.text_tokenizer.convert_token_to_id('</s>')),
                ]
                self.num_tokens += self.num_command_tokens
                self.token_end_id = self.text_tokenizer.convert_token_to_id(
                    '</s>')
            else:
                self.num_command_tokens = 2
                self.num_text_tokens = self.num_tokens - 1
                self._command_tokens = [
                    CommandToken(
                        'pad', '<|endoftext|>',
                        self.text_tokenizer.convert_token_to_id(
                            '<|endoftext|>')),
                    CommandToken(
                        'eos', '<|endoftext|>',
                        self.text_tokenizer.convert_token_to_id(
                            '<|endoftext|>'))
                ]
                self.token_end_id = self.text_tokenizer.convert_token_to_id(
                    '<|endoftext|>')
                if add_block_symbols:
                    if self.tokenizer_model_name.lower().startswith('glm'):
                        unk_token_id = self.num_tokens + 5
                        cls_token_id = self.num_tokens + 2
                        num_tokens_to_add = 5
                    else:
                        unk_token_id = self.text_tokenizer.convert_token_to_id(
                            '<|endoftext|>')
                        cls_token_id = self.text_tokenizer.convert_token_to_id(
                            '<|endoftext|>')
                        num_tokens_to_add = 4
                    self._command_tokens.extend([
                        CommandToken('sop', '<|startofpiece|>',
                                     self.num_tokens),
                        CommandToken('eop', '<|endofpiece|>',
                                     self.num_tokens + 1),
                        CommandToken('cls', '[CLS]', cls_token_id),
                        CommandToken('MASK',
                                     '[MASK]',
                                     self.num_tokens + 3,
                                     lstrip=True),
                        CommandToken('sep', '[SEP]', self.num_tokens + 4),
                        CommandToken('unk', '[UNK]', unk_token_id)
                    ])
                    self.num_tokens += num_tokens_to_add
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

            if self.tokenizer_model_name.lower().startswith('glm'):
                pad_token_id = self.num_tokens
                eos_token_id = self.num_tokens
                unk_token_id = self.num_tokens + 4
            else:
                pad_token_id = self.text_tokenizer.convert_token_to_id('<pad>')
                eos_token_id = self.text_tokenizer.convert_token_to_id('</s>')
                unk_token_id = self.text_tokenizer.convert_token_to_id('<unk>')
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
            self.token_end_id = self.text_tokenizer.convert_token_to_id(
                '</s>')
            if add_block_symbols:
                self._command_tokens.extend([
                    CommandToken('sop', '<|startofpiece|>',
                                 self.num_tokens + 1),
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

    def get_vocab(self):
        return self.text_tokenizer.get_vocab()

    def get_command_id(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name].Id

    def rematch(self, text, tokens):
        """output the mapping relation between raw text and tokenizezd text
        """
        text = text.lower()
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

    @staticmethod
    def _is_control(ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def stem(token):
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_special(ch):
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def _encode(self, text):
        tokens = self.text_tokenizer.tokenize(text)
        ids = self.text_tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def convert_tokens_to_ids(self, tokens):
        res = []
        for token in tokens:
            if token in self.command_token_map:
                res.append(self.command_token_map[token].Id)
            else:
                res.append(self.text_tokenizer.convert_token_to_id(token))
        return res

    def convert_ids_to_tokens(self, ids):
        # if torch.is_tensor(ids):
        #     ids = ids.tolist()
        res = []
        for id in ids:
            if id in self.command_id_map:
                res.append(self.command_id_map[id].token)
            else:
                res.append(self.text_tokenizer.convert_id_to_token(id))
        return self.text_tokenizer.convert_ids_to_tokens(ids)

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
                    tokens.extend(
                        self.text_tokenizer.convert_ids_to_tokens([id]))
                except KeyError:
                    pass
        return self.text_tokenizer.convert_tokens_to_string(
            tokens, self.command_token_map)

    def encode(self, text):
        return self.convert_tokens_to_ids(
            self.text_tokenizer.tokenize(text))

    def decode(self, ids):
        return self.DecodeIds(ids)

    def DecodeTokens(self, tokens):
        """converts wordpiece tokens to a text string"""
        return self.text_tokenizer.convert_tokens_to_string(
            tokens, self.command_token_map)

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
                return self.encode(text)

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

    def CommandTokenIds(self, exception=None):
        result = []
        for s in self._command_tokens:
            if not exception or (exception and s.name not in exception):
                result.append(s.Id)
        return (result)

    def encode_plus_non_glm(
            self,
            text,
            second_text=None,
            truncation=True,
            max_length=None,
    ):

        def get_input_ids(text):
            tokens = self.text_tokenizer.tokenize(text)
            return self.text_tokenizer.convert_tokens_to_ids(tokens)

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(
            second_text) if second_text is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            truncation=truncation,
            max_length=max_length,
        )

    def prepare_for_model(
            self,
            ids: List[int],
            pair_ids: Optional[List[int]] = None,
            add_special_tokens: bool = True,
            truncation: Union[bool, str] = True,
            max_length: Optional[int] = None,
    ):

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}
        total_len = len_ids + len_pair_ids + 3

        # Truncation: Handle max sequence length
        if truncation is True and (max_length is not None
                                   and total_len > max_length):
            self.truncate_sequence(
                max_length,
                ids,
                pair_ids,
                pop_index=-1,
            )

        if add_special_tokens:
            if pair_ids is not None:
                sequence = [self.get_command_id("cls")] + ids + [
                    self.token_end_id
                ] + pair_ids + [self.token_end_id]
                token_type_ids = [0] * (len(ids) + 2) + [1] * (len(pair_ids) +
                                                               1)
            else:
                sequence = [self.get_command_id("cls")
                            ] + ids + [self.token_end_id]
                token_type_ids = [0] * (len(ids) + 2)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] *
                                               len(pair_ids) if pair else [])

        encoded_inputs["input_ids"] = sequence
        encoded_inputs["token_type_ids"] = token_type_ids
        return encoded_inputs

    def encode_plus(  # for Seq2seq
            self,
            source_text: str,
            second_text=None,
            target_text=None,
            truncation=True,
            max_length=None,
            padding=True,
    ):
        if not self.tokenizer_model_name.lower().startswith("glm") and not self.tokenizer_model_name.lower().startswith(
                "alm"):
            return self.encode_plus_non_glm(source_text, second_text,
                                            truncation, max_length)
        sop_id = self.get_command_id('sop')  # start of piece
        eop_id = self.get_command_id('eop')  # end of piece
        sep_id = self.get_command_id('sep')  # seperation

        source_tokens = self.EncodeAsIds(source_text)
        if truncation and max_length:
            self.truncate_sequence(max_length - 2, source_tokens)
        source_tokens = [sop_id] + source_tokens + [sep_id]

        # no pading for consistency
        len_source = len(source_tokens)
        sop_pos = source_tokens.index(sop_id)
        loss_mask = [0] * len_source
        block_position_ids = [0] * len_source
        position_ids = list(range(len_source))

        if target_text:
            target_tokens = self.EncodeAsIds(target_text)
            if max_length:
                target_tokens_length = min(max_length - len(source_tokens), len(target_tokens))
                pad_token = self.get_command_id('pad')
                padding_length = max(0,max_length-len(source_tokens)-target_tokens_length)
                target_tokens = target_tokens[:target_tokens_length] + [pad_token for i in range(padding_length)] + [eop_id]
            else:
                target_tokens += [eop_id]
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

    @staticmethod
    def truncate_sequence(max_length,
                          first_sequence,
                          second_sequence=None,
                          pop_index=-1):

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

    def tokenize_as_tensor(self, texts):
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        sot_token = self.get_command_id('sot')
        eot_token = self.get_command_id('eot')
        return self.text_tokenizer.tokenize(texts,
                                            sot_token=sot_token,
                                            eot_token=eot_token)

    def tokenize(self, text, maxlen=None, add_spatial_tokens=False):
        tokens = self.text_tokenizer.tokenize(text)

        if add_spatial_tokens:
            tokens.insert(0, self.get_command_id('cls'))
            tokens.append(self.get_command_id('sep'))

        if maxlen is not None:
            index = int(self.get_command_id('sep') is not None) + 1
            self.truncate_sequence(maxlen, tokens, pop_index=-index)
        return tokens