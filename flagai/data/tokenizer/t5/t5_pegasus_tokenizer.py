# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.data.tokenizer.bert.wordpiece import BertTokenizer as Tokenizer
from transformers import BertTokenizer
import jieba


class T5PegasusTokenizer(Tokenizer):

    def __init__(self,
                 vocab_path,
                 pre_tokenizer=lambda x: jieba.cut(x, HMM=False),
                 **kwargs):
        super().__init__(vocab_path, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for token in self.pre_tokenizer(text):
            if token in self.vocab:
                split_tokens.append(token)
            else:
                split_tokens.extend(super().tokenize(token))
        return split_tokens

    def encode_plus(
        self,
        text,
        second_text=None,
        add_special_tokens: bool = True,
        truncation=True,
        max_length=None,
    ):
        assert second_text is None, "t5不支持多句子encoding"
        return super().encode_plus(text, second_text, add_special_tokens,
                                   truncation, max_length)


# TODO T5BatchPegasusTokenizer could be mereged into T5PegasusTokenizer
class T5BatchPegasusTokenizer(BertTokenizer):

    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens
