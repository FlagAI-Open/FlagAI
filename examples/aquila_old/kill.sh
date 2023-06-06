#!/bin/bash
#

echo "[INFO] $0: hostfile"
set -u
  hostfile=$1
set +u
app_num=`cat $hostfile |wc -l`
echo "app_num", $app_num


for ((i=1;i<=$app_num;i++ )); do
    ip=`sed -n $i,1p $hostfile|cut -f 1 -d" "`
    echo "ip", $ip
    #ssh $ip "cd $WORKSPACE; bash bmtrain_mgpu.sh $hostfile $configfile $model_name $exp_name $exp_version 1>$LOGFILE 2>&1 &" &
    # ssh $ip "pkill -f '/opt/conda/bin/python -u train_llama_bmtrain_datasets.py'" &
    ssh $ip "killall python"
    #sleep 5
done
