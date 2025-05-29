#!/bin/bash

for alg in "dp" "greedy" "wo_h3t_solver"; do
	bash _run_2080ti.sh 8 1.8b $alg 8
	bash _run_2080ti.sh 8 1.8b $alg 32
done

for alg in "dp" "greedy" "wo_h3t_solver"; do
	bash _run_2080ti.sh 8 6b $alg 128
	bash _run_2080ti.sh 8 6b $alg 512
done
