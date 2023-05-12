# ENVS
# TODO
export FLAGAI_HOME=/data2/gitee/flagai-internal
export WORKSPACE=/data2/workspace_v3/bmt-flashatten-1T

export PYTHONPATH=$FLAGAI_HOME
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_DEBUG=debug
export OMP_NUM_THREADS=4

echo "[INFO] $0: hostfile configfile model_name exp_name exp_version"
set -u
  hostfile=$1
  configfile=$2
  model_name=$3
  exp_name=$4
  exp_version=$5
set +u
# DIST
export HOSTFILE=$hostfile
export CONFIGFILE=$configfile
export NODE_ADDR=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:")
export GPU_NUM_PER_NODE=$(awk -F" |=" '{ranks[$1]=$NF;}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
export NODES_NUM=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
export MASTER_PORT=23456

export TRIGGER_FILE=bmtrain_mgpu_flash.sh

export SCRIPT_FILE=train_llama_bmtrain_datasets-flash-sft.py

## wandb
export WANDB_MODE=offline

## EXP
#export EXP_NAME=llama_7b_8n8g
export EXP_NAME=Aquila-7b-16n8g
export EXP_NAME=Aquila-7b-1n8g
export EXP_NAME=Aquila-7b-6n8g
#export MODEL_NAME=llama-7b-en
export MODEL_NAME=Aquila-7b

export MODEL_NAME=$model_name
export EXP_NAME=$exp_name

export STATE_DICT_DIR=$WORKSPACE/state_dict
export SAVE_DIR=/data2/checkpoints/${EXP_NAME}
export WANDB_DIR=$WORKSPACE/wandb/${EXP_NAME}
export EXP_VERSION_DIR=$SAVE_DIR/$exp_version
mkdir -p $EXP_VERSION_DIR
mkdir -p $WANDB_DIR
## Backup ckpts & scripts into exp versions
# cp -r $STATE_DICT_DIR/$MODEL_NAME $EXP_VERSION_DIR
cp -r $WORKSPACE/$TRIGGER_FILE $EXP_VERSION_DIR
cp -r $hostfile $EXP_VERSION_DIR
cp -r $configfile $EXP_VERSION_DIR

export EPOCH_NUM=1
export BATCH_SIZE=6
export GRADIENT_ACCUM_STEPS=1
export LR=3.0e-4
export LR=1.0e-5
export LR=6.0e-5
export WARMUP_RATE=0.008
export WARMUP_RATE=0.02
export WARMUP_RATE=0.1
export WARMUP_RATE=0.2

## EXTRA OPTS
OPTS=" --batch_size $BATCH_SIZE \
       --epochs $EPOCH_NUM \
       --gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
       --lr $LR \
       --warm_up $WARMUP_RATE \
       --weight_decay 0.1 \
       --adam_beta1 0.9 \
       --adam_beta2 0.95 \
       --save_dir $SAVE_DIR \
       --experiment_name $EXP_NAME \
       --model_name $MODEL_NAME \
       --wandb_dir $WANDB_DIR \
       --yaml_config $CONFIGFILE"

## Trigger job on Each Node when bmt or ddp.
python -u -m torch.distributed.launch \
       --nproc_per_node $GPU_NUM_PER_NODE \
       --nnodes $NODES_NUM \
       --node_rank $RANK \
       --master_addr $MASTER_ADDR \
       --master_port $MASTER_PORT \
       $SCRIPT_FILE \
       --not_call_launch \
       $OPTS

