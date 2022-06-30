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

print_rank_0 = print
"""define some default command tokens for the tokenizer to use"""
token_format = "<{0}>"

COMMAND_TUPLE = namedtuple('CommandToken', ('name', 'token', 'Id'))


def prep_command_tokens(tokenlist, token_format=token_format):
    return [
        CommandToken(tok[0], token_format.format(tok[0]), tok[1])
        for tok in tokenlist
    ]


class CommandToken(object):

    def __init__(self, name, token, Id, lstrip=False, rstrip=False):
        self.name = name
        self.token = token
        self.Id = Id
        self.lstrip = lstrip
        self.rstrip = rstrip

    def __str__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))


DEFAULT_COMMAND_TOKENS = [
    ('pad', 0),
    ('eos', 1),
    ('bos', 2),
    ('unk', 3),
    ('sep', 4),
    ('L2R', 5),
    ('ENC', 6),
    ('MASK', 7),
]
DEFAULT_COMMAND_TOKENS = prep_command_tokens(DEFAULT_COMMAND_TOKENS)
"""define some default type tokens for bert training"""

TYPE_TUPLE = namedtuple('TypeToken', ('name', 'token', 'Id'))


def prep_type_tokens(tokenlist, token_format=token_format):
    return [
        TypeToken(tok[0], token_format.format(tok[0]), tok[1])
        for tok in tokenlist
    ]


class TypeToken(object):

    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(TYPE_TUPLE(self.name, self.token, self.Id))


DEFAULT_TYPE_TOKENS = [
    ('function', 0),
    ('command', 1),
    ('str0', 2),
    ('str1', 3),
    ('str2', 4),
    ('embedding0', 5),
    ('embedding1', 6),
    ('embedding2', 7),
    ('arg0', 8),
    ('arg1', 9),
    ('arg2', 10),
]
DEFAULT_TYPE_TOKENS = prep_type_tokens(DEFAULT_TYPE_TOKENS)





class GLMTokenizer(object):
    """
    Tokenizer object that handles text tokenization, command tokens, and type tokens.

    Command tokens and text tokens are stored together in one mapping of size
    `len(text_tokenizer)+len(command_tokens)`. Command tokens are stored as first
    `len(command_tokens)` tokens. Token idx is stored at `idx+len(command_tokens)`.

    Token types are stored in a separate mapping of size `len(type_tokens)`.
    """

    def __init__(self, text_tokenizer, command_tokens=None, type_tokens=None):
        # set text tokenizer
        self.text_tokenizer = text_tokenizer
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = len(self.text_tokenizer)

        # set command tokens
        if command_tokens is None:
            command_tokens = DEFAULT_COMMAND_TOKENS
        self._command_tokens = command_tokens
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {
            tok.token: tok
            for tok in self._command_tokens
        }
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}
        if not hasattr(self, 'num_command_tokens'):
            self.num_command_tokens = len(self._command_tokens)
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_command_tokens + self.num_text_tokens

        # set type tokens
        if type_tokens is None:
            type_tokens = DEFAULT_TYPE_TOKENS
        self.type_tokens = type_tokens
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}
        if not hasattr(self, 'num_type_tokens'):
            self.num_type_tokens = len(self.type_tokens)

        # parse tokens and vocabs from tokenizer
        self._tokens = list(self.command_token_map.keys()) + list(
            self.text_tokenizer.tokens)
        self._vocab = {t: Id for Id, t in self.command_id_map.items()}
        self._vocab.update({
            t: Id + self.num_command_tokens
            for t, Id in self.text_tokenizer.vocab.items()
        })

        self._text_tokens = list(self.text_tokenizer.tokens)
        self._text_token_vocab = {
            t: Id + self.num_command_tokens
            for t, Id in self.text_tokenizer.vocab.items()
        }

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {
            t: Id
            for Id, t in self.command_id_map.items()
        }

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def __call__(self, text, process_fn=None):
        """run preprocessing and encode text as Ids"""
        return self.EncodeAsIds(text, process_fn=process_fn)

    def __len__(self):
        """total number of tokens"""
        return self.num_tokens

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    def get_type(self, name):
        """get type token corresponding to `name`"""
        return self.type_name_map[name]

    @property
    def tokens(self):
        """list (or iterable) of all tokens for tokenizer"""
        return self._tokens

    @property
    def vocab(self):
        """dictionary mapping tokens to ids for tokenizer"""
        return self._vocab

    @property
    def token_types(self):
        """list (or iterable) of all token types for tokenizer"""
        return self._token_types

    @property
    def token_type_vocab(self):
        """dictionary mapping token types to ids for tokenizer"""
        return self._token_type_vocab

    @property
    def command_tokens(self):
        """list (or iterable) of all command tokens for tokenizer"""
        return self._command_token_tokens

    @property
    def command_token_vocab(self):
        """dictionary mapping command tokens to ids for tokenizer"""
        return self._command_token_vocab

    @property
    def text_tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        return self._text_tokens

    @property
    def text_token_vocab(self):
        """dictionary mapping text tokens to ids for text tokenizer"""
        return self._text_token_vocab

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
        # tokenization = Tokenization(Ids, processed_text, text)
        # tokenization.set_command_tokens(self._command_tokens)
        # return tokenization

    def _encode(self, text):
        raise NotImplementedError

    def EncodeAsTokens(self, text, process_fn=None):
        """
        encode text as tokens using text tokenizer
        """
        tokenization = self.text_tokenizer.EncodeAsTokens(
            text, process_fn=process_fn)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def IdToToken(self, Id, type_token=False):
        """convert Id to token accounting for command and type tokens"""
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id < self.num_command_tokens:
            return self.command_id_map[Id].token
        return self.text_tokenizer.IdToToken(Id - self.num_command_tokens)

    def TokenToId(self, token, type_token=False):
        """convert token to Id accounting for command and type tokens"""
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        if token in self.command_token_map:
            return self.command_token_map[token].Id
        return self.text_tokenizer.TokenToId(token) + self.num_command_tokens

    def DecodeIds(self, Ids, type_token=False):
        """
        convert Ids to tokens accounting for command and type tokens, tokens
        are joined and returned as a string.
        """
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.
                            type_id_map[Id].token for Id in Ids)
        rtn_strs = []
        current_str = []
        for Id in Ids:
            if isinstance(Id, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(Id.token)
            elif Id < self.num_command_tokens:
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(self.command_id_map[Id].token)
            else:
                current_str.append(Id - self.num_command_tokens)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
        return ' '.join(rtn_strs)

    def DecodeTokens(self, Tokens, type_token=False):
        """
        convert tokens to a string accounting for command and type tokens.
        """
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t
                            for t in Tokens)
        rtn_strs = []
        current_str = []
        for t in Tokens:
            if isinstance(t, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t.token)
            elif t in self.command_token_map:
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t)
            else:
                current_str.append(t)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
        return ' '.join(rtn_strs)

import os
from flagai.model.file_utils import _get_model_id, _get_vocab_path
import logging
logger = logging.getLogger(__name__)
class BaseTokenizer(object):
    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        cache_dir=None,
                        tokenizer_flag='sp',
                        *inputs,
                        **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        model_id = _get_model_id(pretrained_model_name_or_path)
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), 'vocabs')
        resolved_vocab_file, resolved_merges_file = None, None
        if tokenizer_flag in ["wp", "bpe"]:
            vocab_file = 'vocab.txt'
            if not os.path.exists(cache_dir + '/' +
                                  vocab_file):
                _get_vocab_path(cache_dir + '/', vocab_file, model_id, rank=0)
            resolved_vocab_file = os.path.join(cache_dir, vocab_file)
        if tokenizer_flag == "bpe":
            merges_file = "vocabs/merges.txt"
            if not os.path.exists(cache_dir + '/' +
                                  merges_file):
                _get_vocab_path(cache_dir + '/', merges_file, model_id, rank=0)
            resolved_merges_file = os.path.join(cache_dir, merges_file)
        tokenizer = cls(resolved_vocab_file,
                        resolved_merges_file,
                        *inputs,
                        **kwargs)
        return tokenizer

    def __init__(self, resolved_vocab_file=None, resolved_vocab_file=None):
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

    def tokenize(self, text):
        if self.do_basic_tokenize:
            split_tokens = []
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(
                    len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens


class Tokenizer(object):
    """
    Tokenizer object that handles text tokenization, command tokens, and type tokens.

    Command tokens and text tokens are stored together in one mapping of size
    `len(text_tokenizer)+len(command_tokens)`. Command tokens are stored as first
    `len(command_tokens)` tokens. Token idx is stored at `idx+len(command_tokens)`.

    Token types are stored in a separate mapping of size `len(type_tokens)`.
    """

    def __init__(self, text_tokenizer):
        # set text tokenizer
        self.text_tokenizer = text_tokenizer
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = len(self.text_tokenizer)

        # parse tokens and vocabs from tokenizer
        self._tokens = list(self.text_tokenizer.tokens)
        self._vocab = {t: Id for t, Id in self.text_tokenizer.vocab.items()}

        self._text_tokens = list(self.text_tokenizer.tokens)
        self._text_token_vocab = {
            t: Id
            for t, Id in self.text_tokenizer.vocab.items()
        }

    def __call__(self, text, process_fn=None):
        """run preprocessing and encode text as Ids"""
        return self.EncodeAsIds(text, process_fn=process_fn)

    def __len__(self):
        """total number of tokens"""
        return self.num_tokens

    @property
    def tokens(self):
        """list (or iterable) of all tokens for tokenizer"""
        return self._tokens

    @property
    def vocab(self):
        """dictionary mapping tokens to ids for tokenizer"""
        return self._vocab

    @property
    def text_tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        return self._text_tokens

    @property
    def text_token_vocab(self):
        """dictionary mapping text tokens to ids for text tokenizer"""
        return self._text_token_vocab

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
        return self.text_tokenizer._tokenize(text)

    def IdToToken(self, Id: int):
        """Token id => token"""
        return self.text_tokenizer._convert_id_to_token(Id)

    def TokenToId(self, token: str):
        """Token => token id"""
        return self.text_tokenizer._convert_token_to_id(token)

    def DecodeIds(self, Ids):
        """A list of token ids => recovered text string"""
        return self.DecodeTokens([self.IdToToken(id) for id in Ids])

    def DecodeTokens(self, tokens):
        """A list of tokens => recovered text string"""
        return self.text_tokenizer.convert_tokens_to_string(tokens)


# class BaseTokenizer(object):

class TextTokenizer(object):
    """
    Interface for text tokenizer
    """

    def __init__(self):
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = 0
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_text_tokens

    def __call__(self, text, process_fn=None):
        return self.EncodeAsIds(text, process_fn)

    def __len__(self):
        return self.num_text_tokens

    @property
    def tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        raise NotImplementedError(
            'TextTokenizer tokens property not implemented')

    @property
    def vocab(self):
        """dictionary mapping tokens to ids"""
        raise NotImplementedError(
            'TextTokenizer vocab property not implemented')

    @staticmethod
    def exists(model_path):
        """check if the filepath for a text tokenizer exists"""
        raise NotImplementedError(
            'TextTokenizer exists method not implemented')

    def Train(self, corpus):
        """train a tokenizer on a data corpus and save model for future use"""
        raise NotImplementedError('TextTokenizer Train not implemented')

    def EncodeAsIds(self, text, process_fn=None):
        """
        Preprocess text and encode as ids. Return a tokenization object with
        original text, processed text, and id tokenization.
        """
        raise NotImplementedError('TextTokenizer EncodeAsIds not implemented')

    def EncodeAsTokens(self, text, process_fn=None):
        """
        Preprocess text and encode as tokens. Return a tokenization object with
        original text, processed text, and token tokenization.
        """
        raise NotImplementedError(
            'TextTokenizer EncodeAsTokens not implemented')

    def IdToToken(self, Id):
        """Convert an Id to Token. Reverse lookup of self.vocab"""
        raise NotImplementedError('TextTokenizer IdToToken not implemented')

    def TokenToId(self, token):
        """Convert a Token to Id. Lookup of self.vocab"""
        raise NotImplementedError('TextTokenizer TokenToId not implemented')

    def DecodeIds(self, Ids):
        """Convert a list or tokenization object of Ids to a text string"""
        raise NotImplementedError('TextTokenizer DecodeIds not implemented')

    def DecodeTokens(self, Tokens):
        """Convert a list or tokenization object of tokens to a text string"""
        raise NotImplementedError('TextTokenizer DecodeTokens not implemented')
