#!/bin/bash
#

set -u
  hostfile=$1
set +u
app_num=`cat $hostfile |wc -l`

WORKSPACE=/data/ldwang/workspace/FlagAI/examples/gpt3_pretrain/llama

for ((i=1;i<=$app_num;i++ )); do
    ip=`sed -n $i,1p $hostfile|cut -f 1 -d" "`
    #scp -r /data/indexed_dataset $ip:/data/ &
    echo "ip", $ip
    ssh $ip "cd $WORKSPACE; rm bmtrain_8n8g.sh; bash bmtrain_mgpu.sh 1>log.txt 2>&1 &" &
    #sleep 5
done
