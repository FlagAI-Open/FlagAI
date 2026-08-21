Question_File=data/gsm8k/gsm8k/main
Sample_File=data/gsm8k/model_answer/llama-3.2-1b-instruct/baseline.jsonl

python evaluation/gsm8k/check_correctness.py \
    --question-file $Question_File \
    --sample-file $Sample_File