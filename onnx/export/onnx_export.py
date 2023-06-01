#!/usr/bin/env python

from PIL import Image
from os import makedirs
from os.path import join
from misc.clip_model import TXT, IMG
from misc.config import ONNX_FP, opset_version, IMG_DIR
from misc.proc import transform, tokenizer
import torch

makedirs(ONNX_FP, exist_ok=True)

JPG = join(IMG_DIR, 'cat.jpg')

image = Image.open(JPG)
image = transform(image)
image = torch.tensor(image)


def onnx_export(outdir, model, args, **kwds):
  name = f'{outdir}.onnx'
  fp = join(ONNX_FP, name)
  torch.onnx.export(
      model,
      args,
      fp,
      export_params=True,
      # verbose=True,
      opset_version=opset_version,
      do_constant_folding=False,
      output_names=['output'],
      **kwds)
  print(name, "DONE\n")
  # rename(fp, join(ONNX_DIR, name))


# 参考 https://github.com/OFA-Sys/Chinese-CLIP/blob/master/cn_clip/deploy/pytorch_to_onnx.py

onnx_export('txt',
            TXT,
            tokenizer(['a photo of cat', 'a image of cat'], ),
            input_names=['input', 'attention_mask'],
            dynamic_axes={
                'input': {
                    0: 'batch',
                    1: 'batch',
                },
                'attention_mask': {
                    0: 'batch',
                    1: 'batch',
                }
            })

onnx_export('img',
            IMG,
            image,
            input_names=['input'],
            dynamic_axes={'input': {
                0: 'batch'
            }})
