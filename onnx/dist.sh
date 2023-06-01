#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

LI="L L-m9 L-m18"

VERSION=$(cat version.txt)

mkdir -p dist

for model in $LI; do
  export MODEL="AltCLIP-XLMR-${model}"
  ./export.sh
  ./export/tar.bz2.py
  mv onnx/$MODEL.tar.bz2 dist/$MODEL-$VERSION.tar.bz2
done
