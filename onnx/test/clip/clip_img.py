#!/usr/bin/env python

from misc.proc import transform
from misc.clip_model import IMG
import torch


def img2vec(img):
  img = transform(img)
  img = torch.tensor(img)
  return IMG.forward(img)


if __name__ == "__main__":
  from misc.config import IMG_DIR
  from misc.norm import norm
  from os.path import join
  from PIL import Image

  fp = join(IMG_DIR, 'cat.jpg')
  img = Image.open(fp)
  vec = img2vec(img)
  print('vec', vec)
  print('norm', norm(vec))
