echo "PYTHONPATH: $PYTHONPATH"
echo "GPU_NUM: $GPU_NUM"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "EPOCH_NUM: $EPOCH_NUM"
echo "EXP_NAME: $EXP_NAME"

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "pwd:" $(pwd)

# configs
#GPU_NUM=$GPU_NUM
#port=29502
#script_file=train_env_xl_bmtrain.py
# ENV_TYPE
#export ENV_TYPE=bmtrain

## TODO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_DEBUG=info
export OMP_NUM_THREADS=4

SCRIPT_FILE=train_env_xl_bmtrain_gpm_xl.py
SAVE_DIR=/share/project/ldwang/checkpoints_${EXP_NAME}

OPTS=" --lr 6.0e-5 \
       --warm_up 0.01 \
       --weight_decay 0.1 \
       --adam_beta1 0.9 \
       --adam_beta2 0.95 \
       --save_dir $SAVE_DIR \
       --experiment_name $EXP_NAME"

python -m torch.distributed.launch \
    --nproc_per_node $GPU_NUM --nnodes $WORLD_SIZE --node_rank $RANK \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    $SCRIPT_FILE --not_call_launch \
    --batch_size $BATCH_SIZE --epochs $EPOCH_NUM \
    --num_gpus $GPU_NUM $OPTS
