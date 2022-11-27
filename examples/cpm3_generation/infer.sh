#! /bin/bash

set -ex

DECODING=beam # options: beam, greedy, random, contrastive
CKPT_PATH=/sharefs/baai-mrnd/xw/cpm3/pytorch_model.pt
INPUT_FILE="/sharefs/webbrain-lijijie/data/CEPSUM/test_public.jsonl"
OUTPUT_FILE="/home/xiangwen/data/output.jsonl"

OPTS=""
OPTS+=" --model-config /sharefs/baai-mrnd/xw/cpm3/config.json" # 配置文件
OPTS+=" --vocab-file /sharefs/baai-mrnd/xw/cpm3/vocab/cpm3/vocab_new.txt" # 词表文件
OPTS+=" --load ${CKPT_PATH}" # 加载checkpoint
OPTS+=" --input-file ${INPUT_FILE}" # 输入文件
# OPTS+=" --span-length 100" # 输出长度
# OPTS+=" --no-repeat-ngram-size 0" # ngram惩罚
# OPTS+=" --repetition-penalty 1" # 重复惩罚
OPTS+=" --no-repeat-ngram-size 0" # ngram惩罚
OPTS+=" --repetition-penalty 1.2" # 重复惩罚
OPTS+=" --output-file ${OUTPUT_FILE}" # 输出文件
OPTS+=" --beam-size 3"
OPTS+=" --max-length 840"
OPTS+=" --span-length 150"
OPTS+=" --top-p 0.9"
OPTS+=" --temperature 0.9"

# if [[ ${DECODING} == "beam" ]]; then
#     # beam search
#     OPTS+=" --beam-size 3"
# if [[ ${DECODING} == "greedy" ]]; then
#     # greedy search
#     OPTS+=" --beam-size 1"
# elif [[ ${DECODING} == "random" ]]; then
#     # top_k, top_p sampling
#     OPTS+=" --beam-size 1"
#     OPTS+=" --temperature 0.9"
#     OPTS+=" --top-k 0"
#     OPTS+=" --top-p 0.9"
#     OPTS+=" --random-sample"
# elif [[ ${DECODING} == "contrastive" ]]; then
#     # contrastive search
#     OPTS+=" --beam-size 1"
#     OPTS+=" --use-contrastive-search"
# else
#     echo "${DECODING} is a not a valid decoding strategy!"
# fi

export PYTHONIOENCODING=utf-8
CMD="python3 infer.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee infer.log

