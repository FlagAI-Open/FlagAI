#! /bin/bash

set -ex

MASTER_ADDR=127.0.0.1
MASTER_PORT=8080
NNODES=1
NODE_RANK=0

GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

TRAIN_BATCH=4
LR=1e-2
WARM_UP=100
CKPT_STEP=28500
DATASET=lcsts

OPTS=""
OPTS+=" --model-config /sharefs/baai-mrnd/xw/cpm3/config.json"
OPTS+=" --vocab-file /sharefs/baai-mrnd/xw/cpm3/vocab/cpm3/vocab_new.txt"
OPTS+=" --batch-size ${TRAIN_BATCH}"
OPTS+=" --train-iters 10000"
OPTS+=" --save-iters 500"
OPTS+=" --save-name tuning-checkpoint"
OPTS+=" --max-length 512"
OPTS+=" --save /sharefs/baai-mrnd/xw/model_save/${DATASET}_batch$[${TRAIN_BATCH} * ${GPUS_PER_NODE}]_lr${LR}_warm${WARM_UP}_ckpt${CKPT_STEP}_${TUNING}"
OPTS+=" --lr ${LR}"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters ${WARM_UP}"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 4.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step 0"
OPTS+=" --load /sharefs/webbrain-lijijie/transformers_models/cpm-3/pytorch_model.pt"
OPTS+=" --eval-step 50"
OPTS+=" --eval-batch-size 50"
OPTS+=" --dataset ${DATASET}"
OPTS+=" --log-dir /sharefs/baai-mrnd/xw/cpm3_log/${DATASET}_batch$[${TRAIN_BATCH} * ${GPUS_PER_NODE}]_lr${LR}_warm${WARM_UP}_ckpt${CKPT_STEP}_${TUNING}"
OPTS+=" --epochs 5"

export PYTHONIOENCODING=utf-8
CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} /home/xiangwen/FlagAI-internal/examples/cpm3_finetune/finetune_cpm3.py ${OPTS}"
echo ${CMD}

if [[ $NODE_RANK == 0 ]]; then
    ${CMD} 2>&1 | tee /sharefs/baai-mrnd/xw/cpm3_log/${DATASET}_batch$[${TRAIN_BATCH} * ${GPUS_PER_NODE}]_lr${LR}_warm${WARM_UP}_ckpt${CKPT_STEP}_${TUNING}.log.5epoch
else
    ${CMD}
fi
