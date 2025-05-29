from datasets import load_dataset
from utils import clean_answer, is_correct
import argparse
import json


def main(args):
    print('test correctness of file:', args.sample_file)
    
    questions = load_dataset(args.question_file)
    if questions.get('test', None) is not None:
        questions = questions['test']
    else:
        questions = questions['train']

    raw_answer = []
    with open(args.sample_file, 'r') as f:
        for line in f:
            raw_answer.append(json.loads(line)['choices'][0]['turns'][0])


    result = []
    for ra, question in zip(raw_answer, questions):
        pred_ans = clean_answer(ra)
        is_cor = is_correct(pred_ans, question['answer'])
        result.append(is_cor)
    
    print('Accuracy:', sum(result) / len(result))


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "--question-file",
        type=str,
        default='data/gsm8k/gsm8k/main'
    )
    args.add_argument(
        "--sample-file",
        type=str,
        default='data/gsm8k/model_answer/llama-3-8b-instruct/baseline.jsonl'
    )
    args = args.parse_args()
    main(args)
