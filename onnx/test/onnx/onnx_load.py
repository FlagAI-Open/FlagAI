#!/usr/bin/env python

import onnxruntime
from misc.config import ONNX_FP
from os.path import join

session = onnxruntime.SessionOptions()
option = onnxruntime.RunOptions()
option.log_severity_level = 2


def onnx_load(kind):
  fp = join(ONNX_FP, f'{kind}.onnx')

  sess = onnxruntime.InferenceSession(fp,
                                      sess_options=session,
                                      providers=[
                                          'TensorrtExecutionProvider',
                                          'CUDAExecutionProvider',
                                          'CoreMLExecutionProvider',
                                          'CPUExecutionProvider'
                                      ])
  return sess
