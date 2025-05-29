#!/bin/bash

for alg in "dp" "greedy" "wo_h3t_solver"; do
	bash _run_a100.sh 8 6b $alg 128
	bash _run_a100.sh 8 6b $alg 512
done

for alg in "dp" "greedy" "wo_h3t_solver"; do
	bash _run_a100.sh 8 13b $alg 128
	bash _run_a100.sh 8 13b $alg 512
done
