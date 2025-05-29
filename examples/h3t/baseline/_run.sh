#!/bin/bash
set -ex

DS_CONFIG=ds_config.json

TP=1
PP=1

N_GPU=$1
MODEL_SIZE=$2
GLOBAL_BATCH=$3
ZERO_STAGE=$4
PREFIX=$5

echo "N_GPU = ${N_GPU} ; MODEL_SIZE = ${MODEL_SIZE} ; GLOBAL_BATCH = ${GLOBAL_BATCH} ; ZERO_STAGE = ${ZERO_STAGE}" >> result.log.txt

if [[ "$MODEL_SIZE" == "1.8b" ]]; then
	NLAYERS=48
	HIDDEN=1024
	ATTN_HEAD=32
	FFN_HIDDEN=16384
elif [[ "$MODEL_SIZE" == "6b" ]]; then
	NLAYERS=48
	HIDDEN=1024
	ATTN_HEAD=128
	FFN_HIDDEN=65536
elif [[ "$MODEL_SIZE" == "13b" ]]; then
	NLAYERS=48
	HIDDEN=2048
	ATTN_HEAD=128
	FFN_HIDDEN=65536
elif [[ "$MODEL_SIZE" == "100b" ]]; then
	NLAYERS=56
	HIDDEN=12288
	ATTN_HEAD=96
	FFN_HIDDEN=49152
fi



MICRO_BATCH=$((GLOBAL_BATCH/N_GPU))

OFFLOAD_DEVICE="cpu"

OUTPUT_DIR=ds_z${ZERO_STAGE}_${MODEL_SIZE}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
mkdir -p $OUTPUT_DIR

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "offload_optimizer": {
      "device": "$OFFLOAD_DEVICE",
      "buffer_count": 4,
      "pipeline_read": false,
      "pipeline_write": false,
      "pin_memory": true
    }
  },

  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

export NCCL_DEBUG=warn 

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"


${PREFIX} deepspeed pretrain_bert.py \
    --bert-no-binary-head \
    --cpu-optimizer \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $ATTN_HEAD \
    --ffn-hidden-size $FFN_HIDDEN \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 64 \
    --lr 1.0e-4 \
    --min-lr 6.0e-6 \
    --lr-decay-style constant \
    --log-interval 32 \
    --vocab-file bert-vocab.txt \
    --checkpoint-activations \
    --fp16 \
    --tensorboard-dir $OUTPUT_DIR \
    $ds_args \
    --exit-interval 64 | tee ${OUTPUT_DIR}/output.log

date >> result.log.txt
echo "" >> result.log.txt
