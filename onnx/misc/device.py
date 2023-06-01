#!/usr/bin/env python

import torch
import os

device = os.getenv('DEVICE')
if not device:
  if torch.cuda.is_available():
    device = 'cuda'
  elif torch.backends.mps.is_available():
    device = 'mps'
  else:
    device = 'cpu'

DEVICE = torch.device(device)
