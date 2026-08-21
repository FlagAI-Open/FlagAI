#!/bin/bash

export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin
export MACA_CLANG=/opt/maca/mxgpu_llvm
export DEVINFO_ROOT=/opt/maca
export WCUDA_PATH=/opt/maca/tools/wcuda
export CUDA_PATH=/opt/maca/tools/wcuda
export MACA_SMALL_PAGESIZE_ENABLE=1
export MALLOC_THRESHOLD=99
export MCPYTORCH_DISABLE_PRINT=1
export MCCL_NET_GDR_LEVEL=7
export MCCL_MAX_NCHANNELS=16
export MCCL_P2P_LEVEL=SYS
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export FORCE_ACTIVATE_WAIT=1
export MHA_USE_BLAS=ON
#export MHA_BWD_NO_ATOMIC_F64=1
export LD_LIBRARY_PATH=/root/FWD_76_BWD_79:/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib
#export LD_LIBRARY_PATH=/mnt/plin/blas0419:/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib
export SET_DEVICE_NUMA_PREFERRED=1
export MAX_JOBS=20
export PYTORCH_ENABLE_SAME_RAND_A100=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
FlagScale=Megatron-LM-FlagScale_new
export PYTHONPATH=/mnt/baai/ldwang/$FlagScale/megatron:/mnt/baai/ldwang/$FlagScale
export GLOO_SOCKET_IFNAME=eth0
#export MACA_CACHE_DISABLE=1
#export JIT_SAVE_TEMPS_DIR=1
#export MXLOG_LEVEL=debug
#ulimit -c unlimited
#ulimit -a

git config --global --add safe.directory /mnt/baai/ldwang/$FlagScale
cd /mnt/baai/ldwang/$FlagScale

# Distributed training variables
NNODES=${WORLD_SIZE}
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_REAL_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=${RANK}

# Parallelism variables
TP=4
PP=8
EP=1
DP=$((${GPU_NUM}/${TP}/${PP}))

# Paths
BASE_PATH=/mnt/baai/ldwang
# TODO Datasets
source /mnt/baai/ldwang/FlagScale/examples/aquila/datasets_mx.sh
DATA_PATH=$DATASET_K74_BETA
TOKENIZER_PATH=/mnt/baai/ldwang/FlagScale/examples/aquila/1_8B/tokenizer_hf

EXPNAME=K74_BETA_Merge
EXPNAME=K74_BETA_Single
EXPNAME=K74_BETA_Single_Lin
TASK_NAME=ldwang_moe_8x16b_pretrain_new_WS${WORLD_SIZE}_TP${TP}_PP${PP}_EP${EP}_DP${DP}_${EXPNAME} 
OUTPUTS_DIR=${BASE_PATH}/${TASK_NAME}/node${NODE_RANK}

#CHECKPOINTS_DIR=${OUTPUTS_DIR}/checkpoints
LOGS_DIR=${OUTPUTS_DIR}/logs
LOG_PATH=${OUTPUTS_DIR}/logs/${NODE_RANK}_${POD_NAME}.log
mkdir -p ${OUTPUTS_DIR}/logs
LOGS_PIDS_DIR=${OUTPUTS_DIR}/logs/pids
LOGS_DETAILS_DIR=${OUTPUTS_DIR}/logs/details
TENSORBOARD_DIR=${OUTPUTS_DIR}/tensorboard
WANDB_DIR=${OUTPUTS_DIR}/wandb

CHECKPOINTS_DIR="/mnt1/ckpt/${TASK_NAME}"
## Init Load
#    --finetune \
#    Native Load \
#LOAD_CHECKPOINTS_DIR=/mnt1/baai/ldwang/checkpoints/Aquila_16B_8x16b
#LOAD_CHECKPOINTS_DIR=/mnt1/baai/ldwang/checkpoints/ldwang_dense_1x16b_pretrain_new_WS128_TP2_PP2_EP1_DP256_b2b_K74_BETA/iter_0096000_8x16b
#LOAD_CHECKPOINTS_DIR=/mnt1/ckpt/ldwang_moe_8x16b_pretrain_new_WS128_TP4_PP8_EP1_DP32_K74_BETA_Single
## Resume
LOAD_CHECKPOINTS_DIR=$CHECKPOINTS_DIR

SRC_PATH=flagscale/train/train_mixtral.py

mkdir -p ${OUTPUTS_DIR}
mkdir -p ${CHECKPOINTS_DIR}
mkdir -p ${LOGS_PIDS_DIR}
mkdir -p ${LOGS_DETAILS_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${WANDB_DIR}

MAX_SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=4096


#    --lr-decay-style cosine \
#    --lr 1.485221e-04 \
#    --lr-warmup-samples 614400 \
NETWORK_SIZE_ARGS=" \
    --use-mcore-models \
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 20480 \
    --num-attention-heads 40 \
    --seq-length ${MAX_SEQ_LEN} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-05 \
    --group-query-attention \
    --num-query-groups 8 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --rotary-base 1000000.0 \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --untie-embeddings-and-output-weights \
    --use-flash-attn \
    --init-method-std 0.02 \
    --num-experts 8 \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 0.001 \
    --seed 2024 \
    --micro-batch-size 1 \
    --global-batch-size 3072 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --weight-decay 0.1 \
    --train-samples 122880000 \
    --lr-decay-style linear \
    --lr 1.475556e-04 \
    --min-lr 2e-05 \
    --lr-warmup-samples 7372800 \
    --data-path ${DATA_PATH} \
    --reset-position-ids \
    --reset-attention-mask \
    --split 1 \
    --tokenizer-path ${TOKENIZER_PATH} \
    --tokenizer-type QwenTokenizer \
    --vocab-size 151851 \
    --make-vocab-size-divisible-by 64 \
    --distributed-timeout-minutes 30 \
    "

CMD=" \
    dlrover-run \
    --rdzv-id 20240410_074637.966007 \
    --network-check --exclude-straggler --save-at-breakpoint \
    --max_restarts=3  --rdzv_conf pend_timeout=1800 \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${GPUS_PER_NODE} \
    --log_dir ${LOGS_DETAILS_DIR} \
    --redirects 3 \
    --tee 3 \
    ${SRC_PATH} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --bf16 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --log-interval 1 \
    --wandb-project $TASK_NAME \
    --wandb-exp-name $TASK_NAME \
    --wandb-save-dir $WANDB_DIR \
    --tensorboard-dir $TENSORBOARD_DIR \
    --override-opt_param-scheduler \
    --save-interval 50 \
    --save ${CHECKPOINTS_DIR} \
    --load ${LOAD_CHECKPOINTS_DIR} \
    --use-mcore-models \
    ${NETWORK_SIZE_ARGS} \
    "

echo ${CMD}
${CMD} 2>&1 | tee ${LOG_PATH}; exit ${PIPESTATUS[0]}
