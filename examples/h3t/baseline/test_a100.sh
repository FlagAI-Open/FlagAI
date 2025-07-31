#!/bin/bash

for zero_level in "3" "2"; do
	bash _run.sh 8 6b 128 $zero_level
	bash _run.sh 8 6b 512 $zero_level

	bash _run.sh 8 13b 128 $zero_level
	bash _run.sh 8 13b 512 $zero_level
done

