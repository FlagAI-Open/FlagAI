# coding=utf-8

import os
from typing import Union
import torch
import bmtrain as bmt
from model_center.utils import check_web_and_convert_path

class BaseTokenizer:
    """
    The current implementation is mainly to adapt the training framework of the Transformers toolkit, 
    and replace the original model implementation.
    TODO we will change to our SAM implementation in the future, which will be a more efficient tokenizer
    """
    def __init__(self, tokenizer_type):
        self.tokenizer_type = tokenizer_type

    def from_pretrained(self, pretrained_model_name_or_path: Union[str, os.PathLike], *args, **kwargs):
        pretrained_model_name_or_path = check_web_and_convert_path(pretrained_model_name_or_path, 'tokenizer')
        return self.tokenizer_type.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
