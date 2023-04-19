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

exp_YYYYMMDDHH=$(date +"%Y%m%d%H")
echo "exp_YYYYMMDDHH", $exp_YYYYMMDDHH

export PROJ_HOME=/data/ldwang
WORKSPACE=/data/ldwang/workspace/FlagAI/examples/gpt3_pretrain/llama
SAVE_DIR=$PROJ_HOME/checkpoints/${exp_name}/$exp_YYYYMMDDHH
LOGFILE=$SAVE_DIR/$configfile.log.txt
echo "LOGFILE", $LOGFILE

for ((i=1;i<=$app_num;i++ )); do
    ip=`sed -n $i,1p $hostfile|cut -f 1 -d" "`
    echo "ip", $ip
    ssh $ip "cd $WORKSPACE; mkdir -p $SAVE_DIR; bash bmtrain_mgpu.sh $hostfile $configfile $model_name $exp_name $exp_YYYYMMDDHH 1>$LOGFILE 2>&1 &" &
    #ssh $ip "pkill -f '/opt/conda/bin/python -u train_llama_bmtrain_datasets.py'" &
    #ssh $ip "killall python"
    #sleep 5
done
