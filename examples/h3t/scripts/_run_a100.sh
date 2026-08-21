#!/bin/bash

log_file="result.log.txt"

world_size=$1
model_size=$2
alg_name=$3
batch_size=$4
prefix=$5

port=10086

for ((i=0;;++i)); do
	port=$((port^1))
	cmd="${prefix} torchrun --nproc_per_node=${world_size} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:${port} run.py --model-size ${model_size} --alg-name ${alg_name} --batch-size ${batch_size}"
	echo $cmd
	$cmd
	if (($?==0)); then
		break
	fi
done
echo $cmd >> ${log_file}
date >> ${log_file}
echo "" >> ${log_file}

