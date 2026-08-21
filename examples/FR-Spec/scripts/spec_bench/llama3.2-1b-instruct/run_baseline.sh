export CUDA_VISIBLE_DEVICES=2
Model_Path=meta-llama/Llama-3.2-1B-instruct
Model_id="llama-3.2-1b-instruct"
Bench_name="spec_bench"

python3 evaluation/inference_baseline.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id $Model_id/baseline \
    --memory-limit 0.8 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "llama-3"
