#!/bin/bash

log_file="result.log.txt"

nnodes=$1
node_rank=$2
nproc_per_node=8
if ((node_rank==0)); then
	is_host=1
else
	is_host=0
fi

model_size=$3
alg_name=$4
batch_size=$5
prefix=$6

addr= # Addr
port=10086

cmd="${prefix} torchrun --max_restarts=1000 --nnodes=${nnodes} --node_rank=${node_rank} --rdzv_conf=is_host=${is_host} --nproc_per_node=${nproc_per_node} --rdzv_id=11382 --rdzv_backend=c10d --rdzv_endpoint=${addr}:${port} run.py --model-size ${model_size} --alg-name ${alg_name} --batch-size ${batch_size}"
echo $cmd
$cmd

if ((node_rank==0)); then
	echo $cmd >> ${log_file}
	date >> ${log_file}
	echo "" >> ${log_file}
fi

