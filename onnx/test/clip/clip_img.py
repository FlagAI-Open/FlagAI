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
  from os.path import join
  fp = join(IMG_DIR, 'cat.jpg')
  from PIL import Image
  img = Image.open(fp)
  print(img2vec(img))
