#! /bin/bash

if [[ $DLS_TASK_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
else
    MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
    MASTER_ADDR="${MASTER_HOST%%:*}"
    MASTER_PORT="${MASTER_HOST##*:}"
    NNODES="$DLS_TASK_NUMBER"
    NODE_RANK="$DLS_TASK_INDEX"
fi

GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

OPTS=""
OPTS+=" --model-config /sharefs/baai-mrnd/xw/cpm3/config.json"
OPTS+=" --vocab-file /home/xiangwen/CPM-3/训练与微调/vocab/cpm3/vocab_new.txt"
OPTS+=" --batch-size 4"
OPTS+=" --train-iters 200000"
OPTS+=" --save-iters 500"
OPTS+=" --save-name noam-0.1-0.01-checkpoint"
OPTS+=" --max-length 2048"
OPTS+=" --save /sharefs/baai-mrnd/xw/model_save/"
OPTS+=" --lr 0.1"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 4.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step 0"
OPTS+=" --log-dir /sharefs/baai-mrnd/xw/cpm3_log/"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} /home/xiangwen/FlagAI-internal/examples/cpm3_pretrain/pretrain_cpm3.py ${OPTS}"
echo ${CMD}

if [[ $NODE_RANK == 0 ]]; then
    ${CMD} 2>&1 | tee /mnt/sfs_turbo/zz/CPM-3/src/logs/pretrain/cpm3-6b-final-noam-0.1-0.01.log.all
else
    ${CMD}
fi
