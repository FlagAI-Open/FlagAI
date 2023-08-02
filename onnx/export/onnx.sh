#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

./down.py
./onnx_export_processor.py
./onnx_export.py
