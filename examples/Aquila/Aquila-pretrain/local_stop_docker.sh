#!/bin/bash
#
# Defined by user
export PROJ_HOME=$PWD

echo "[INFO] $0: hostfile"
set -u
  hostfile=$1
set +u
NODES_NUM=`cat $hostfile |wc -l`
echo "NODES_NUM": $NODES_NUM
if [ $NODES_NUM -ne 1 ];then
    echo "Make Sure One Node in hostfile"
    exit 0
fi

killall python
