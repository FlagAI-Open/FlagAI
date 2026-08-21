Question_File=data/human_eval/question.jsonl
Sample_File=data/human_eval/model_answer/llama-3-8b-instruct/baseline.jsonl

python evaluation/he_local/check_correctness.py \
    --question-file $Question_File \
    --sample-file $Sample_File