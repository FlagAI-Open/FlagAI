#!/usr/bin/env python

from misc.proc import tokenizer
from onnx_load import onnx_load

MODEL = onnx_load('txt')


def txt2vec(li):
  text, attention_mask = tokenizer(li)
  text = text.numpy()
  attention_mask = attention_mask.numpy()
  output = MODEL.run(None, {'input': text, 'attention_mask': attention_mask})
  return output[0]


if __name__ == '__main__':
  from test_txt import TEST_TXT
  for li in TEST_TXT:
    r = txt2vec(li)
    for txt, i in zip(li, r):
      print(txt)
      print(i)
      print('\n')
