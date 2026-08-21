#!/bin/bash

for zero_level in "3" "2"; do
	bash _run.sh 8 1.8b 128 $zero_level
	bash _run.sh 8 1.8b 512 $zero_level
    
	bash _run.sh 8 6b 8 $zero_level
	bash _run.sh 8 6b 32 $zero_level
done