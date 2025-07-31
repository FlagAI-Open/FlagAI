export CUDA_VISIBLE_DEVICES=7
Model_Path=meta-llama/Meta-Llama-3-8B-Instruct
Eagle_Path=models/LLaMA3-Instruct-8B-FR-Spec
Model_id="llama-3-8b-instruct"
Bench_name="gsm8k"


python3 -m pdb evaluation/inference_eagle.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}/eagle-original \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "llama-3" \
    --max-new-tokens 512