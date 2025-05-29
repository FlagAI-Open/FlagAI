#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/cdgm0705/hyx/calora-master" 
VERSION="3b"
DATASET="WiC"
SAVE_NAME="WiC-base" # to change

OPTS=""
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config /cdgm0705/hyx/models/t5-3b" 
OPTS+=" --batch-size 16"
OPTS+=" --train-iters 1500"
OPTS+=" --save-iters 1000"
OPTS+=" --max-encoder-length 512"
OPTS+=" --max-decoder-length 4"
OPTS+=" --save /cdgm0705/hyx/calora-master/results/${DATASET}-${SAVE_NAME}"
OPTS+=" --save-name ${DATASET}-${SAVE_NAME}"
OPTS+=" --lr 0.001"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 70"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 128"
OPTS+=" --pet True"
OPTS+=" --pet-init-type random"
OPTS+=" --recover False"
OPTS+=" --distill False"
OPTS+=" --model-ckpt-path /cdgm0705/hyx/models/t5-3b/pytorch_model.pt"


CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/t5/finetune_t5_superglue.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1  | tee /cdgm0705/hyx/calora-master/checkpoints/logs/t5_glue/${DATASET}-${SAVE_NAME}.log
