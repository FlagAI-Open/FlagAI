Question_File=data/human_eval/question.jsonl
Sample_File=data/human_eval/model_answer/llama-3.2-1b-instruct/baseline.jsonl

python evaluation/human_eval/check_correctness.py \
    --question-file $Question_File \
    --sample-file $Sample_File