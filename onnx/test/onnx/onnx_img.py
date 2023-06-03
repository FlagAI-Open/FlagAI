#!/usr/bin/env python

from misc.proc import transform
from PIL import Image
from onnx_load import onnx_load

IMG = onnx_load('Img')


def img2vec(img):
  return IMG.run(None, {'input': transform(img)})[0]


if __name__ == '__main__':
  from misc.config import IMG_DIR
  from os.path import join
  img = Image.open(join(IMG_DIR, 'cat.jpg'))

  img_data = transform(img)
  import torch
  print('img data size', torch.tensor(img_data).size())

  vec = img2vec(img)
  print('vec', vec)
  IMG_NORM = onnx_load('ImgNorm')
  print('norm', IMG_NORM.run(None, {'input': img_data})[0])
