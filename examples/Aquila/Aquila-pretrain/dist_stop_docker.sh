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

for ((i=1;i<=$NODES_NUM;i++ )); do
    ip=`sed -n $i,1p $hostfile|cut -f 1 -d" "`
    echo "ip": $ip
    ssh $ip "killall python"
    #sleep 5
done
