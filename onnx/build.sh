#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

GIT=$(cat ../.git/config | grep url | awk -F= '{print $2}' | sed -e 's/[ ]*//g')

# 替换 : 为 /
GIT=${GIT//://}

# 替换 git@ 为 https://
GIT=${GIT//git@/https:\/\/}

docker build --build-arg GIT=$GIT . -t altclip-onnx
