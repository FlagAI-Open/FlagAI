#!/usr/bin/env python

from misc.proc import transform
from PIL import Image
from onnx_load import onnx_load

MODEL = onnx_load('img')


def img2vec(img):
  return MODEL.run(None, {'input': transform(img)})[0]


if __name__ == '__main__':
  from misc.config import IMG_DIR
  from os.path import join
  img = Image.open(join(IMG_DIR, 'cat.jpg'))
  vec = img2vec(img)
  print(vec)
