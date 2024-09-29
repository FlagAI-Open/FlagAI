#!/bin/bash
# ---------------------------------------------------------------
# [Author]       : shixiaofeng, ldwang
# [Descriptions] :

set -x -e
RUN_ID=$(date +"%Y%m%d%H")

PROJ_HOME=/share/project/ldwang/text-quality

GOLDEN_MODEL=qwen2

FREEZE_BACKBONE=1

PRETRAINED_MODEL=bge-m3-xlmroberta-nodrop
MODEL_PATH=/share/project/ldwang/Aquila3/bge-m3-nodrop
MAX_SEQ_LEN=2048

DATA_VERSION=145k
EVAL_SAMPLES=15000
train_file=/share/projset/ldwang/text-quality/datasets/$DATA_VERSION/train.jsonl
validation_file=/share/projset/ldwang/text-quality/datasets/$DATA_VERSION/val.jsonl

# Final Settings
BSZ=256
EPOCHNUM=20
LR=1e-4

EXP=quality_scorer_${PRETRAINED_MODEL}-fb$FREEZE_BACKBONE
SUBEXP=from-${GOLDEN_MODEL}-ds$DATA_VERSION-ep$EPOCHNUM-bsz$BSZ-lr$LR-seq$MAX_SEQ_LEN
WANDB_RUN_ID=$EXP-$SUBEXP-$RUN_ID

output_path=$PROJ_HOME/checkpoints/$EXP/$SUBEXP/$RUN_ID

RUN_SCRIPT=run_classification.py
mkdir -p $output_path
cp $0 $output_path/
cp $RUN_SCRIPT $output_path/

MASTER_PORT=20001
export CUDA_VISIBLE_DEVICES="0,1,2,3"
GPU_NUM=4
MICRO_BSZ=1
GRAD_ACCUM=$(( $BSZ / $GPU_NUM / $MICRO_BSZ ))
export WANDB_MODE="offline"
export WANDB_PROJECT="data_quality_scorer"
export WANDB_DIR=$output_path
export WANDB_RUN_ID=$WANDB_RUN_ID

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPU_NUM \
    --master_addr=127.0.0.1 \
    --master_port=$MASTER_PORT \
    run_classification.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $output_path \
    --overwrite_output_dir \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --shuffle_train_dataset \
    --metric_name mse \
    --text_column_name text \
    --label_column_name label \
    --do_regression true \
    --do_train \
    --do_eval \
    --freeze_backbone $FREEZE_BACKBONE \
    --model_name $PRETRAINED_MODEL \
    --max_eval_samples $EVAL_SAMPLES \
    --max_seq_length ${MAX_SEQ_LEN} \
    --per_device_train_batch_size $MICRO_BSZ \
    --per_device_eval_batch_size $MICRO_BSZ \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --pad_to_max_length False \
    --save_strategy epoch \
    --save_only_model \
    --evaluation_strategy epoch \
    --seed 42 \
    --lr_scheduler_type cosine \
    --learning_rate $LR \
    --num_train_epochs $EPOCHNUM \
    --logging_steps 1 \
    --bf16 \
    --run_name $WANDB_RUN_ID \
    --report_to wandb 1>>$output_path/run.log 2>&1 &

