#!/usr/bin/env python

from misc.proc import tokenizer
from misc.clip_model import TXT


def txt2vec(li):
  return TXT.forward(*tokenizer(li))


if __name__ == "__main__":
  from os.path import join
  from glob import glob
  from misc.config import ROOT
  from test_txt import TEST_TXT
  from misc.norm import norm

  li = glob(join(ROOT, 'jpg/*.jpg'))
  for li in TEST_TXT:
    r = txt2vec(li)
    for txt, vec in zip(li, r):
      print(txt)
      print('vec', vec)
      print('norm', norm(vec))
      print('\n')
