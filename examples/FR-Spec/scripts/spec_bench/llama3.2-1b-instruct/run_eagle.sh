export CUDA_VISIBLE_DEVICES=3
Model_Path=meta-llama/Llama-3.2-1B-instruct
Eagle_Path=models/LLaMA3.2-Instruct-1B-FR-Spec
Model_id="llama-3.2-1b-instruct"
Bench_name="spec_bench"

python3 evaluation/inference_eagle.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id $Model_id/eagle-original \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "llama-3" \
    --eagle-num-iter 3 \
    --eagle-tree-size 30
