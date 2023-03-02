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
from typing import List

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased':
    "flagai/data/tokenizer/bert/vocabs/bert-base-uncased-vocab.txt",
    'bert-large-uncased':
    "flagai/data/tokenizer/bert/vocabs/bert-large-uncased-vocab.txt",
    'bert-base-cased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}

from ..tokenizer import Tokenizer
from .wordpiece import BertTokenizer
from ..tokenizer import CommandToken

class BertWordPieceTokenizer(Tokenizer):
    """
    Loads a pretrained WordPiece tokenizer from `cache_dir` for tokenization
    in BERT training. Default to bert-large-uncased tokenizer.
    """

    def __init__(self, tokenizer_model_type=None, cache_dir=None):
        # default to bert-large-uncased tokenizer
        if tokenizer_model_type not in PRETRAINED_VOCAB_ARCHIVE_MAP:
            tokenizer_model_type = 'bert-large-uncased'
        if not torch.distributed.is_initialized(
        ) or torch.distributed.get_rank() == 0:
            print('loading GLMBertWordPieceTokenizer (', tokenizer_model_type,
                  ') from cache_dir ', cache_dir)
        do_lower_case = not ('-cased' in tokenizer_model_type
                             or 'chinese' in tokenizer_model_type)

        self.text_tokenizer = BertTokenizer.from_pretrained(
            tokenizer_model_type,
            do_lower_case=do_lower_case,
            cache_dir=cache_dir)
        if not torch.distributed.is_initialized(
        ) or torch.distributed.get_rank() == 0:
            print('loaded', tokenizer_model_type)
        # disable max len warnings by increasing max len
        self.text_tokenizer.max_len = int(1e12)

        # # parse tokens and vocabs from tokenizer
        self._tokens = list(self.text_tokenizer.vocab.keys())
        self._vocab = {k: v for k, v in self.text_tokenizer.vocab.items()}
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
        return self.text_tokenizer.vocab[getattr(self.text_tokenizer, "_token_" + str(token))]

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    def _encode(self, text):
        tokens = self.text_tokenizer.tokenize(text)
        ids = self.text_tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def encode_plus(
        self,
        text,
        second_text=None,
        add_special_tokens: bool = True,
        truncation=True,
        max_length=None,
    ):

        def get_input_ids(text):
            tokens = self.text_tokenizer.tokenize(text)
            return self.text_tokenizer.convert_tokens_to_ids(tokens)

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(
            second_text) if second_text is not None else None

        return self.text_tokenizer.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
        )

    def rematch(self, text, tokens):
        return self.text_tokenizer.rematch(text, tokens)

    def decode(
        self,
        token_ids: List[int],
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> str:

        filtered_tokens = self.text_tokenizer.convert_ids_to_tokens(token_ids)

        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            current_sub_text.append(token)

        if current_sub_text:
            sub_texts.append(
                self.text_tokenizer.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.text_tokenizer.clean_up_tokenization(text)
            return clean_text
        else:
            return text

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
        token = token.lower()
        try:
            return self.text_tokenizer.vocab[token]
        except KeyError:
            try:
                return self.text_tokenizer.vocab[token.upper()]
            except KeyError:
                return self.text_tokenizer.vocab[token.strip()]

    def DecodeIds(self, Ids):
        """converts ids to wordpiece tokens and joins them as a text string"""
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

    def DecodeTokens(self, Tokens):
        """converts wordpiece tokens to a text string"""
        return ' '.join(Tokens)
