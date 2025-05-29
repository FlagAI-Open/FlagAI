#!/bin/bash

for zero_level in "3" "2"; do
    bash _run_multinode.sh 8 13b 1024 $zero_level
    bash _run_multinode.sh 8 13b 4096 $zero_level

    bash _run_multinode.sh 8 100b 1024 $zero_level
    bash _run_multinode.sh 8 100b 2048 $zero_level
done
