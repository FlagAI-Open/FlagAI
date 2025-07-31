#!/bin/bash

node_rank=$1

for alg in "dp" "greedy" "wo_h3t_solver"; do
	bash _run_8nodes_a100.sh 8 $node_rank 13b $alg 1024
	bash _run_8nodes_a100.sh 8 $node_rank 13b $alg 4096
done

for alg in "dp" "greedy" "wo_h3t_solver"; do
	bash _run_8nodes_a100.sh 8 $node_rank 100b $alg 1024
	bash _run_8nodes_a100.sh 8 $node_rank 100b $alg 2048
done

