#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex
mkdir -p out dist model

NAME=altclip-onnx

if ! [ -z $ORG ]; then
  NAME=$ORG/$NAME
fi

if [ -z "$1" ]; then
  exe="exec bash"
  it="-it"
else
  exe="exec bash -c \"$@\""
  it=""
fi

docker run \
  -v $DIR/misc:/app/misc \
  -v $DIR/img:/app/img \
  -v $DIR/export:/app/export \
  -v $DIR/onnx:/app/onnx \
  -v $DIR/dist:/app/dist \
  -v $DIR/model:/app/model \
  -v $DIR/test:/app/test \
  $it --rm $NAME bash -c "$exe"
