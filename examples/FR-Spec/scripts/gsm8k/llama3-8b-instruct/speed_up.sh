tokenizer_path="meta-llama/Meta-Llama-3-8B-Instruct"
Vocab=32768

baseline="data/gsm8k/model_answer/llama-3-8b-instruct/baseline.jsonl"
eagle_original="data/gsm8k/model_answer/llama-3-8b-instruct/eagle-original.jsonl"
eagle_fr_spec="data/gsm8k/model_answer/llama-3-8b-instruct/eagle-fr-spec-$Vocab.jsonl"

echo "EAGLE ORIGINAL"
python evaluation/human_eval/speed.py \
    --file-path $eagle_original \
    --base-path $baseline \
    --checkpoint-path $tokenizer_path

echo "EAGLE FR-SPEC"
python evaluation/human_eval/speed.py \
    --file-path $eagle_fr_spec \
    --base-path $baseline \
    --checkpoint-path $tokenizer_path