#! /bin/bash

set -ex

DECODING=beam # options: beam, greedy, random, contrastive
CKPT_PATH=/home/shunxing1234/model/cpm3/pytorch_model.pt
INPUT_FILE=/home/shunxing1234/data/cpm3/CPM3-eval/outgen/test.jsonl
OUTPUT_FILE=/home/shunxing1234/data/temp.txt

OPTS=""
OPTS+=" --model-config /home/shunxing1234/model/CPM-3/推理/src/config/cpm3-large-32head.json" # 配置文件
OPTS+=" --vocab-file /home/shunxing1234/model/CPM-3/推理/vocab/cpm3/vocab_new.txt" # 词表文件
OPTS+=" --load ${CKPT_PATH}" # 加载checkpoint
OPTS+=" --input-file ${INPUT_FILE}" # 输入文件
# OPTS+=" --span-length 400" # 输出长度
OPTS+=" --no-repeat-ngram-size 0" # ngram惩罚
OPTS+=" --repetition-penalty 1" # 重复惩罚
#OPTS+=" --no-repeat-ngram-size 0" # ngram惩罚
##OPTS+=" --repetition-penalty 1.2" # 重复惩罚
OPTS+=" --output-file ${OUTPUT_FILE}" # 输出文件
OPTS+=" --beam-size 1"
OPTS+=" --use-contrastive-search"
OPTS+=" --max-length 840"
OPTS+=" --span-length 150"
OPTS+=" --top-p 0.9"


#if [[ ${DECODING} == "beam" ]]; then
#    OPTS+=" --beam-size 3"
#if [[ ${DECODING} == "greedy" ]]; then
#    OPTS+=" --beam-size 1"
#elif [[ ${DECODING} == "random" ]]; then
#    OPTS+=" --beam-size 1"
#    OPTS+=" --temperature 0.9"
#    OPTS+=" --top-k 0"
#    OPTS+=" --top-p 0.9"
#    OPTS+=" --random-sample"
#elif [[ ${DECODING} == "contrastive" ]]; then
#    OPTS+=" --beam-size 1"
#    OPTS+=" --use-contrastive-search"
#else
#    echo "${DECODING} is a not a valid decoding strategy!"
#fi

export PYTHONIOENCODING=utf-8
CMD="python3 infer.py ${OPTS}"
echo ${CMD}
${CMD} 2>&1 | tee infer.log