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

import itertools
from .glm_10b_en_tokenizer import GLM10bENTokenizer
import regex as re

from ..tokenizer import GLMTokenizer, CommandToken, TypeToken


class GLM10bENBPETokenizer(GLMTokenizer):

    def __init__(self,
                 tokenizer_model_type="gpt2",
                 add_block_symbols=False,
                 add_task_mask=False,
                 add_decoder_mask=False):
        """
        Args:
            tokenizer_model_type: (str):
                The type of the tokenizer
            add_block_symbols: (bool)
                When add_block_symbol is True, a bunch of block-masking-related special tokens will be added to the vocab
            add_task_mask (bool)
                when add_task_mask is True, the generation mask token and gap sentence mask token will be distinguished
            add_decoder_mask: (bool)
                When add_decoder_mask is True, some tokens of the block spans will be masked for BERT, and a special token
                    for that will be added to vocab
        """
        self.text_tokenizer = GLM10bENTokenizer.from_pretrained(
            tokenizer_model_type)

        # disable max len warnings by increasing max len
        self.text_tokenizer.max_len = int(1e12)
        self.num_tokens = len(self.text_tokenizer.encoder)
        self.num_type_tokens = 2
        if tokenizer_model_type.startswith('roberta'):
            self.num_command_tokens = 6
            self.num_text_tokens = self.num_tokens - 3
            self._command_tokens = [
                CommandToken('pad', '<|endoftext|>',
                             self.text_tokenizer.encoder['</s>']),
                CommandToken('eos', '<|endoftext|>',
                             self.text_tokenizer.encoder['</s>']),
                CommandToken('sep', '[SEP]',
                             self.text_tokenizer.encoder['</s>']),
                CommandToken('cls', '[CLS]',
                             self.text_tokenizer.encoder['<s>']),
                CommandToken('MASK',
                             '[MASK]',
                             self.text_tokenizer.encoder['<mask>'],
                             lstrip=True),
                CommandToken('unk', '[UNK]',
                             self.text_tokenizer.encoder['<unk>'])
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
                             self.text_tokenizer.encoder['<|endoftext|>']),
                CommandToken('eos', '<|endoftext|>',
                             self.text_tokenizer.encoder['<|endoftext|>'])
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
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {
            tok.token: tok
            for tok in self._command_tokens
        }
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}

        self.type_tokens = [
            TypeToken('str0', '<str0>', 0),
            TypeToken('str1', '<str1>', 1),
        ]
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}

        self._tokens = list(self.text_tokenizer.encoder.keys())
        self._vocab = {k: v for k, v in self.text_tokenizer.encoder.items()}

        self._text_tokens = list(self._tokens)
        self._text_token_vocab = {
            k: v
            for k, v in self.text_tokenizer.encoder.items()
        }

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {
            t: Id
            for Id, t in self.command_id_map.items()
        }

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

        for idx, tok in self.command_id_map.items():
            self.text_tokenizer.decoder[idx] = tok.token

    def EncodeAsIds(self, text, process_fn=None):
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
            """
            tok_list contains a list of CommandTokens
            text is the original text string
            """
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
                    else:  # 感觉永远不会进到这里
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (self.text_tokenizer.encode(token)
                     if token not in self._command_token_tokens else
                     [self.command_token_map[token].Id]
                     for token in tokenized_text)))

        no_split_tokens = self._command_tokens
        Ids = split_on_tokens(no_split_tokens, processed_text)
        return Ids
        # tokenization = Tokenization(Ids, processed_text, text)
        # tokenization.set_command_tokens(self._command_tokens)
        # return tokenization

    def _encode(self, text):
        return self.text_tokenizer.encode(text)

    def EncodeAsTokens(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = []
        for token in re.findall(self.text_tokenizer.pat, processed_text):
            token = ''.join(self.text_tokenizer.byte_encoder[b]
                            for b in token.encode('utf-8'))
            tokens.extend(
                bpe_token
                for bpe_token in self.text_tokenizer.bpe(token).split(' '))
        return tokens
        # tokenization = Tokenization(tokens, processed_text, text, asIds=False)
        # tokenization.set_command_tokens(self._command_tokens)
        #
        # return tokenization

    def DecodeAsTokens(self, Ids):
        return [self.IdToToken(x) for x in Ids]

    def IdToToken(self, Id, type_token=False):
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id in self.command_id_map:
            return self.command_id_map[Id].token
        return self.text_tokenizer.decoder[Id]

    def TokenToId(self, token, type_token=False):
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        bpe_token = self.EncodeAsTokens(token)[0]
        return self.text_tokenizer.encoder[bpe_token]
        # return

    def DecodeIds(self, Ids, type_token=False):
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.
                            type_id_map[Id].token for Id in Ids)
        # if isinstance(Ids, Tokenization):
        #     Ids = Ids.tokenization
        return self.text_tokenizer.decode(Ids)

    def DecodeTokens(self, Tokens, type_token=False):
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t
                            for t in Tokens)
        # if isinstance(Tokens, Tokenization):
        #     Tokens = Tokens.tokenization
        return self.text_tokenizer.decode(
            [self.TokenToId(tok) for tok in Tokens])

    def tokens_to_text(self, tokens):
        return self.text_tokenizer.sp_model.decode(tokens)

    def ids_to_text(self, ids):
        tokens = self.DecodeIds(ids)
        return self.text_tokenizer.sp_model.decode(tokens)
