# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from __future__ import absolute_import, division, print_function, unicode_literals
from typing import List, Union, Optional
import collections
import logging
import os
import unicodedata
from io import open
import requests

logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased':
    "bert-base-uncased-vocab.txt",
    'bert-large-uncased':
    "bert-large-uncased-vocab.txt",
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
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
}

VOCAB_ARCHIVE_URLS_MAP = {
    'bert-base-uncased':
    "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    'bert-large-uncased':
    "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt",
    'bert-base-chinese':
    "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt"
}

VOCAB_NAME = 'vocab.txt'


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 max_len=None,
                 do_basic_tokenize=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BertTokenizer.
        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          do_lower_case: Whether to lower case the input
                         Only has an effect when do_wordpiece_only=False
          do_basic_tokenize: Whether to do basic tokenization before wordpiece.
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
          never_split: List of tokens which will never be split during tokenization.
                         Only has an effect when do_wordpiece_only=False
        """
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

        self._token_pad = '[PAD]'
        self._token_cls = '[CLS]'
        self._token_sep = '[SEP]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'

        for token in ['pad', 'cls', 'sep', 'unk', 'mask']:
            _token_id = self.vocab[getattr(self, "_token_" + str(token))]
            setattr(self, "_token_" + str(token) + "_id", _token_id)

        self.token_start_id = self._token_cls_id
        self.token_end_id = self._token_sep_id
        self.token_unk_id = self._token_unk_id

    def tokenize(self, text, maxlen=None, add_spatial_tokens=False):
        if self.do_basic_tokenize:
            split_tokens = []
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)

        if add_spatial_tokens:
            split_tokens.insert(0, self._token_cls)
            split_tokens.append(self._token_sep)

        if maxlen is not None:
            index = int(self._token_sep is not None) + 1
            self.truncate_sequence(maxlen, split_tokens, pop_index=-index)

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

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        cache_dir=None,
                        *inputs,
                        **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """

        #
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:  # 如果pretrained_model_name_or_path是名字，找到对应的路径
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[
                pretrained_model_name_or_path]
        else:
            vocab_file = pretrained_model_name_or_path
        # if not os.path.exists(vocab_file) and pretrained_model_name_or_path in VOCAB_ARCHIVE_URLS_MAP:

        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)

        # redirect to the cache, if necessary
        try:
            if cache_dir is None:
                cache_dir = os.path.join(os.path.dirname(__file__), 'vocabs')
            # TODO: the logic of file cache is not clear
            resolved_vocab_file = os.path.join(cache_dir, vocab_file)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    vocab_file))
            return None
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[
                pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)

        if not os.path.exists(resolved_vocab_file):  # 如果没有这个vocab文件， 那我们就要下载
            if pretrained_model_name_or_path in VOCAB_ARCHIVE_URLS_MAP:
                url = VOCAB_ARCHIVE_URLS_MAP[pretrained_model_name_or_path]
                f = requests.get(url)
                os.makedirs(os.path.dirname(resolved_vocab_file),
                            exist_ok=True)
                with open(resolved_vocab_file, "wb") as code:
                    code.write(f.content)

        # Instantiate tokenizer.
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
        return tokenizer

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
                sequence = [self.token_start_id] + ids + [
                    self.token_end_id
                ] + pair_ids + [self.token_end_id]
                token_type_ids = [0] * (len(ids) + 2) + [1] * (len(pair_ids) +
                                                               1)
            else:
                sequence = [self.token_start_id] + ids + [self.token_end_id]
                token_type_ids = [0] * (len(ids) + 2)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] *
                                               len(pair_ids) if pair else [])

        encoded_inputs["input_ids"] = sequence
        encoded_inputs["token_type_ids"] = token_type_ids
        return encoded_inputs

    def encode_plus(
        self,
        text,
        second_text=None,
        add_special_tokens: bool = True,
        truncation=True,
        max_length=None,
    ):

        def get_input_ids(text):
            tokens = self.tokenize(text)
            return self.convert_tokens_to_ids(tokens)

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(
            second_text) if second_text is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
        )

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

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
        Args:
            out_string (`str`): The text to clean up.
        Returns:
            `str`: The cleaned-up string.
        """
        out_string = (out_string.replace(" .", ".").replace(" ?", "?").replace(
            " !", "!").replace(" ,", ",").replace(" ' ", "'").replace(
                " n't",
                "n't").replace(" 'm", "'m").replace(" 's", "'s").replace(
                    " 've", "'ve").replace(" 're", "'re"))
        return out_string

    def decode(
        self,
        token_ids: List[int],
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> str:

        filtered_tokens = self.convert_ids_to_tokens(token_ids)

        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            # if token in self.added_tokens_encoder:
            #     if current_sub_text:
            #         sub_texts.append(self.convert_tokens_to_string(current_sub_text))
            #         current_sub_text = []
            #     sub_texts.append(token)
            # else:
            current_sub_text.append(token)

        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
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
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def truncate_sequence(self,
                          max_length,
                          first_sequence: List[str],
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

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                # token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
