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

BASE_PATH="/local/apps/calora" 
VERSION="3b"
DATASET="MRPC"
SAVE_NAME="$1-inherit-recover-distill" # to change

mkdir -p /data/checkpoints/results/${DATASET}-${SAVE_NAME}

OPTS=""
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config /mnt/data/user/tc_agi/user/zhaoweilin/t5-${VERSION}" 
OPTS+=" --batch-size 16"
OPTS+=" --train-iters 1500"
OPTS+=" --save-iters 1000"
OPTS+=" --max-encoder-length 128"
OPTS+=" --max-decoder-length 4"
OPTS+=" --save /data/checkpoints/results/${DATASET}-${SAVE_NAME}"
OPTS+=" --save-name ${DATASET}-${SAVE_NAME}"
OPTS+=" --lr 0.0005"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 1000"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 10.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --comp-type $1"
OPTS+=" --pet True"
OPTS+=" --pet-init-type inherit"
OPTS+=" --recover True"
OPTS+=" --distill True"
OPTS+=" --quant-ckpt-path /mnt/data/user/tc_agi/user/zhaoweilin/gongbt/BMCook/BMCook-new-config/bmcook/results/t5-3b-test-q/checkpoint.pt"
OPTS+=" --moe-ckpt-path /mnt/data/user/tc_agi/user/zhaoweilin/zhangzhengyan/param_split/" 
OPTS+=" --pr-ckpt-path /mnt/data/user/tc_agi/user/zhaoweilin/gongbt/BMCook/BMCook-new-config/bmcook/results/t5-3b-test-d-p/checkpoint.pt"
OPTS+=" --spr-ckpt-path /mnt/data/user/tc_agi/user/zhaoweilin/gongbt/BMCook/BMCook-new-config/bmcook/results/t5-3b-test-d-sp/checkpoints/ckpt-100000.pt"
OPTS+=" --model-ckpt-path /mnt/data/user/tc_agi/user/zhaoweilin/t5-3b/pytorch_model.pt"
OPTS+=" --mix-ckpt-path /mnt/data/user/tc_agi/user/zhaoweilin/gongbt/BMCook/BMCook-new-config/bmcook/results/t5-3b-test-q-d-p/checkpoint.pt"
OPTS+=" --inherit-ckpt-path /data/checkpoints/results/MRPC-MRPC/MRPC-MRPC17.pt" # to change
OPTS+=" --quant-config-path ${BASE_PATH}/examples/t5/quant_config.json"
OPTS+=" --pr-config-path ${BASE_PATH}/examples/t5/prune_config.json"
OPTS+=" --spr-config-path ${BASE_PATH}/examples/t5/sprune_config.json"
OPTS+=" --mix-layer-ckpt-path /mnt/data/user/tc_agi/user/zhaoweilin/zhangzhengyan/moe-3b-qdp/param_split/"


CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/t5/finetune_t5_glue.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1  | tee /data/checkpoints/logs/t5_glue/${DATASET}-${SAVE_NAME}.log
