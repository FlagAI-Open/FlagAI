echo "PYTHONPATH: $PYTHONPATH"
echo "GPU_NUM: $GPU_NUM"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "EPOCH_NUM: $EPOCH_NUM"
echo "SCRIPT_FILE: $SCRIPT_FILE"

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "pwd:" $(pwd)

# configs
#GPU_NUM=$GPU_NUM
#port=29502
#script_file=train_env_xl_bmtrain.py

python -m torch.distributed.launch \
    --nproc_per_node $GPU_NUM --nnodes $WORLD_SIZE --node_rank $RANK \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    $SCRIPT_FILE --not_call_launch \
    --batch_size $BATCH_SIZE --epochs $EPOCH_NUM \
    --num_gpus $GPU_NUM
