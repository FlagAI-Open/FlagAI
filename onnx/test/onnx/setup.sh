#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

pip install -r ./requirements.txt
if [ -x "$(command -v nvidia-smi)" ]; then
  pip install onnxruntime-gpu
else
  system=$(uname)
  arch=$(uname -m)
  if [[ $system == *"Darwin"* ]] && [[ $arch == "arm64" ]]; then
    pip install onnxruntime-silicon
  else
    grep 'vendor_id' /proc/cpuinfo | awk '{print $3}' | grep 'GenuineIntel' >/dev/null
    if [ $? -eq 0 ]; then
      echo 'Intel CPU'
      pip install onnxruntime-openvino
    else
      echo 'Non-Intel CPU'
      pip install onnxruntime
    fi
  fi
fi
