# ENVS
export FLAGAI_HOME=/data/ldwang/workspace/FlagAI
export PYTHONPATH=$FLAGAI_HOME
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_DEBUG=debug
export OMP_NUM_THREADS=4

# DIST
export GPU_NUM_PER_NODE=8
export NODES_NUM=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=23456

export SCRIPT_FILE=train_llama_bmtrain.py
export SCRIPT_FILE=train_llama_bmtrain_datasets.py

## wandb
export WANDB_MODE=offline

## EXP
export EXP_NAME=llama_7b_1n8g
export EXP_NAME=llama_7b_1n8g_new_data
export MODEL_NAME=llama-7b-en
export SAVE_DIR=/data/ldwang/checkpoints/${EXP_NAME}
export WANDB_DIR=/data/ldwang/wandb/${EXP_NAME}
mkdir -p $SAVE_DIR
mkdir -p $WANDB_DIR
export EPOCH_NUM=1
export BATCH_SIZE=4

## EXTRA OPTS
OPTS=" --batch_size $BATCH_SIZE \
       --epochs $EPOCH_NUM \
       --lr 6.0e-5 \
       --warm_up 0.01 \
       --weight_decay 0.1 \
       --adam_beta1 0.9 \
       --adam_beta2 0.95 \
       --save_dir $SAVE_DIR \
       --experiment_name $EXP_NAME \
       --model_name $MODEL_NAME \
       --wandb_dir $WANDB_DIR"

## Trigger job on Each Node when bmt or ddp.
python -m torch.distributed.launch \
       --nproc_per_node $GPU_NUM_PER_NODE \
       --nnodes $NODES_NUM \
       --node_rank $RANK \
       --master_addr $MASTER_ADDR \
       --master_port $MASTER_PORT \
       $SCRIPT_FILE \
       --not_call_launch \
       $OPTS
