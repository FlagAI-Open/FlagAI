#!/usr/bin/env python

from .config import PROCESS_DIR
from transformers import CLIPProcessor


class AltCLIPProcess(CLIPProcessor):
  tokenizer_class = ("XLMRobertaTokenizer", "XLMRobertaTokenizerFast")

  def __init__(self, feature_extractor, tokenizer):
    super().__init__(feature_extractor, tokenizer)


proc = AltCLIPProcess.from_pretrained(PROCESS_DIR)

_tokenizer = proc.tokenizer
_transform = proc.feature_extractor


def transform(img):
  img = _transform(img)
  return img['pixel_values']


def tokenizer(li):
  tokenizer_out = _tokenizer(li,
                             padding=True,
                             truncation=True,
                             max_length=77,
                             return_tensors='pt')
  text = tokenizer_out["input_ids"]
  attention_mask = tokenizer_out["attention_mask"]
  return text, attention_mask
