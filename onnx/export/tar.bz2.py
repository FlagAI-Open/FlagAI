#!/usr/bin/env python

from misc.config import ONNX_DIR

import bz2
import tarfile


def txz(folder_path, output_path):
    stream = bz2.BZ2File(output_path, 'w')

    with tarfile.TarFile(fileobj=stream, mode='w') as tar:
        tar.add(folder_path, arcname='')

    stream.close()


txz(ONNX_DIR, ONNX_DIR + '.tar.bz2')
