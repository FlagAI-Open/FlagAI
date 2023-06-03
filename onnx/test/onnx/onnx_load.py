#!/usr/bin/env python

import onnxruntime
from misc.config import ONNX_FP
from os.path import join
import os

session = onnxruntime.SessionOptions()
option = onnxruntime.RunOptions()
option.log_severity_level = 2


def onnx_load(kind):
  fp = join(ONNX_FP, f'{kind}.onnx')
  provider = os.getenv('ONNX_PROVIDER')
  providers = [provider] if provider else None
  sess = onnxruntime.InferenceSession(fp,
                                      sess_options=session,
                                      providers=providers)
  return sess


if __name__ == '__main__':
  from onnxruntime import get_all_providers
  print('all providers :\n%s\n' % get_all_providers())
  sess = onnx_load('Txt')
  providers = sess.get_providers()
  print('now can use providers :\n%s\n' % providers)
