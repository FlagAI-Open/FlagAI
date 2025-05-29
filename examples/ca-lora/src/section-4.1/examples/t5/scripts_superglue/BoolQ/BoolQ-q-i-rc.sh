#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/local/apps/calora"  
VERSION="3b"
DATASET="BoolQ"
SAVE_NAME="quant-inherit-recover" # to change

OPTS=""
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config /mnt/data/user/tc_agi/user/zhaoweilin/t5-${VERSION}" 
OPTS+=" --batch-size 32"
OPTS+=" --train-iters 800"
OPTS+=" --save-iters 1000"
OPTS+=" --max-encoder-length 512"
OPTS+=" --max-decoder-length 4"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --save-name ${DATASET}-${SAVE_NAME}"
OPTS+=" --lr 0.001"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 100"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 10.0"
OPTS+=" --loss-scale 128"
OPTS+=" --pet True"
OPTS+=" --comp-type quant"
OPTS+=" --pet-init-type inherit"
OPTS+=" --recover True"
OPTS+=" --distill False"
OPTS+=" --quant-ckpt-path /yourpath/t5-3b/pytorch_model.pt"
OPTS+=" --moe-ckpt-path /yourpath/param_split/" 
OPTS+=" --pr-ckpt-path /yourpath/BMCook/BMCook-new-config/bmcook/results/t5-3b-test-d-p/checkpoint.pt"
OPTS+=" --spr-ckpt-path /yourpath/BMCook/BMCook-new-config/bmcook/results/t5-3b-test-d-sp/checkpoints/ckpt-100000.pt"
OPTS+=" --model-ckpt-path /yourpath/.cache/model_center/t5-3b/pytorch_model.pt"
OPTS+=" --mix-ckpt-path /yourpath/BMCook/BMCook-new-config/bmcook/results/t5-3b-test-q-d-p/checkpoint.pt"
OPTS+=" --inherit-ckpt-path /yourpath/calora/inherit/BoolQ.pt" # to change
OPTS+=" --quant-config-path ${BASE_PATH}/examples/t5/quant_config.json"
OPTS+=" --pr-config-path ${BASE_PATH}/examples/t5/prune_config.json"
OPTS+=" --spr-config-path ${BASE_PATH}/examples/t5/sprune_config.json"
OPTS+=" --mix-layer-ckpt-path /mnt/data/user/tc_agi/user/zhaoweilin/zhangzhengyan/moe-3b-qdp/param_split/"


CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/t5/finetune_t5_superglue.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1  | tee /data/checkpoints/logs/t5_superglue/${DATASET}-${SAVE_NAME}.log
