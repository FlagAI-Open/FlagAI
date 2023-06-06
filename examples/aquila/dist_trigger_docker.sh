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

exp_version=$(date +"%Y%m%d%H%M")
echo "exp_version", $exp_version

# TODO
export WORKSPACE=/share/project/64node-bmt-flashatten
SAVE_DIR=$WORKSPACE/${exp_name}
EXP_VERSION_DIR=$SAVE_DIR/$exp_version
mkdir -p $EXP_VERSION_DIR
LOGFILE=$EXP_VERSION_DIR/$configfile.log.txt
echo "LOGFILE", $LOGFILE

for ((i=1;i<=$app_num;i++ )); do
    ip=`sed -n $i,1p $hostfile|cut -f 1 -d" "`
    echo "ip", $ip
    ssh $ip "cd $WORKSPACE; bash bmtrain_mgpu.sh $hostfile $configfile $model_name $exp_name $exp_version 1>$LOGFILE 2>&1 &" &
    # ssh $ip "pkill -f '/opt/conda/bin/python -u train_llama_bmtrain_datasets.py'" &
    #ssh $ip "killall python"
    #sleep 5
done
