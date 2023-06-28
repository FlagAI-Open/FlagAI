#!/bin/bash
#
# Defined by user
export PROJ_HOME=$PWD

echo "[INFO] $0: hostfile configfile model_name exp_name"
set -u
  hostfile=$1
  configfile=$2
  model_name=$3
  exp_name=$4
set +u
NODES_NUM=`cat $hostfile |wc -l`
echo "NODES_NUM": $NODES_NUM
if [ $NODES_NUM -ne 1 ];then
    echo "Make Sure One Node in hostfile"
    exit 0
fi

exp_YYYYMMDDHH=$(date +"%Y%m%d%H")
echo "exp_YYYYMMDDHH": $exp_YYYYMMDDHH

SAVE_DIR=$PROJ_HOME/checkpoints_out/${exp_name}/$exp_YYYYMMDDHH
LOGFILE=$SAVE_DIR/$configfile.log.txt
echo "LOGFILE": $LOGFILE

cd $PROJ_HOME;
mkdir -p $SAVE_DIR;
bash bmtrain_mgpu.sh $hostfile $configfile $model_name $exp_name $exp_YYYYMMDDHH 1>$LOGFILE 2>&1 &
