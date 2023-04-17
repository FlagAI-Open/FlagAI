#!/bin/bash
#

echo "[INFO] $0: hostfile configfile model_name exp_name"
set -u
  hostfile=$1
  configfile=$2
  model_name=$3
  exp_name=$4
set +u
app_num=`cat $hostfile |wc -l`
echo "app_num", $app_num

WORKSPACE=/data/ldwang/workspace/FlagAI/examples/gpt3_pretrain/llama
export SAVE_DIR=/data/ldwang/checkpoints/${exp_name}
mkdir -p $SAVE_DIR/configs
LOGFILE=$SAVE_DIR/configs/$configfile.log.txt
echo "LOGFILE", $LOGFILE

for ((i=1;i<=$app_num;i++ )); do
    ip=`sed -n $i,1p $hostfile|cut -f 1 -d" "`
    echo "ip", $ip
    ssh $ip "cd $WORKSPACE; bash bmtrain_mgpu.sh $hostfile $configfile $model_name $exp_name 1>$LOGFILE 2>&1 &" &
    #ssh $ip "pkill -f '/opt/conda/bin/python -u train_llama_bmtrain_datasets.py'" &
    #ssh $ip "killall python"
    #sleep 5
done
