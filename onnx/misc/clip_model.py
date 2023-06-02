#!/usr/bin/env python

import torch
from misc.norm import norm
import torch.nn as nn
from .device import DEVICE
from .config import MODEL_FP
from flagai.model.mm.AltCLIP import CLIPHF

MODEL = CLIPHF.from_pretrained(MODEL_FP)

MODEL.eval()
MODEL.to(DEVICE)


class Img(nn.Module):

  def __init__(self):
    super(Img, self).__init__()
    self.model = MODEL

  def forward(self, image):
    with torch.no_grad():
      image = image.to(DEVICE)
      return self.model.get_image_features(image)


class ImgNorm(Img):

  def forward(self, image):
    return norm(super(ImgNorm, self).forward(image))


class Txt(nn.Module):

  def __init__(self):
    super(Txt, self).__init__()
    self.model = MODEL

  def forward(self, text, attention_mask):
    with torch.no_grad():
      text = text.to(DEVICE)
      attention_mask = attention_mask.to(DEVICE)
      return self.model.get_text_features(text, attention_mask=attention_mask)


class TxtNorm(Txt):

  def forward(self, text, attention_mask):
    return norm(super(TxtNorm, self).forward(text, attention_mask))


IMG = Img()
IMG_NORM = ImgNorm()

TXT = Txt()
TXT_NORM = TxtNorm()
