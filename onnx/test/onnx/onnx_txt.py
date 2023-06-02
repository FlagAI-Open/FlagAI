#!/usr/bin/env python

from misc.proc import tokenizer
from onnx_load import onnx_load

TXT = onnx_load('Txt')


def txt2vec(li):
  text, attention_mask = tokenizer(li)
  text = text.numpy()
  attention_mask = attention_mask.numpy()
  output = TXT.run(None, {'input': text, 'attention_mask': attention_mask})
  return output[0]


if __name__ == '__main__':

  from test_txt import TEST_TXT

  TXT_NORM = onnx_load('TxtNorm')

  for li in TEST_TXT:
    r = txt2vec(li)
    text, attention_mask = tokenizer(li)
    text = text.numpy()
    attention_mask = attention_mask.numpy()
    output = TXT_NORM.run(None, {
        'input': text,
        'attention_mask': attention_mask
    })[0]
    for txt, vec, norm in zip(li, r, output):
      print(txt)
      print('vec', vec)
      print('norm', norm)
      print('\n')
