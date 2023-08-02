#!/usr/bin/env python

from misc.config import ONNX_DIR
from os.path import basename
import bz2
import tarfile


def txz(src, to):
  stream = bz2.BZ2File(to, 'w')

  with tarfile.TarFile(fileobj=stream, mode='w') as tar:
    tar.add(src, arcname=basename(src))

  stream.close()


txz(ONNX_DIR, ONNX_DIR + '.tar.bz2')
