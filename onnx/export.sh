#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

cmd="export MODEL=$MODEL;export/onnx.sh"

if ! [ -z $1 ]; then
  cmd="$cmd && $@"
fi

./bash.sh $cmd
