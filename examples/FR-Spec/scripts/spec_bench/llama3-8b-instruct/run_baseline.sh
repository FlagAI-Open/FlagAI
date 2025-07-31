export CUDA_VISIBLE_DEVICES=0
Model_Path=meta-llama/Meta-Llama-3-8B-Instruct
Model_id="llama-3-8b-instruct"
Bench_name="spec_bench"

python3 evaluation/inference_baseline.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id $Model_id/baseline \
    --memory-limit 0.8 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "llama-3"
