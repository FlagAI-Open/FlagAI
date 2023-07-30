import argparse
import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from flagai.auto_model.auto_loader import AutoLoader
import random
import numpy as np
from flagai.model.predictor.predictor import Predictor
from flagai.model.predictor.aquila import aquila_generate
import torch 

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, default='./gsm8k_test.txt', help='Output file for claude-instant')
parser.add_argument('--eval_only', action='store_true', help='Only evaluate the model')

gsm8k = load_dataset('gsm8k', 'main')
gsm8k_test = gsm8k['test']

def load():
    state_dict = "./checkpoints_in"
    model_name = 'aquilachat-7b'
    device = "cuda:1"

    loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True,
    fp16=True)

    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    model.eval()
    model.half()
    model.to(device)

    return model, tokenizer

def parse_answer_file(answer_file):
    lines = open(answer_file, 'r').readlines()

    accuracy = 0
    last_number = 0
    should_find_answer = True
    should_find_reference_answer = False

    for i, l in enumerate(lines):
        try:
            if should_find_answer:
                last_number = re.findall(r'\d+', l)[-1]
        except:
            pass

        if should_find_reference_answer and l.startswith('####'):
            reference_answer = l.split('####')[1].strip()
            if reference_answer == last_number:
                accuracy += 1
        elif l.startswith('===== CASE'):
            should_find_answer = True
            should_find_reference_answer = False
        elif l.startswith('Reference Answer'):
            should_find_answer = False
            should_find_reference_answer = True

    print('Accuracy: ', accuracy / len(gsm8k_test['question']) * 100)

def main(args):

    if args.eval_only:
        parse_answer_file(args.output_file)
        return

    run_count = 0
    model, tokenizer = load()

    with open(args.output_file, 'w') as f:
        for q, a in tqdm(zip(gsm8k_test['question'], gsm8k_test['answer']), total=len(gsm8k_test['question'])):

            run_count += 1
            
            with torch.no_grad():
                response = aquila_generate(tokenizer, model, 
                                               prompts=[q], max_gen_len=2048, 
                                               )

            cleaned_response = response.strip()
            
            f.write(f'===== CASE {run_count} =====\n')
            f.write(f'Question\n: {q}\n')
            f.write(f'Claude-instant Answer\n: {cleaned_response}\n')
            f.write(f'Reference Answer\n: {a}\n\n')

            run_count += 1

    parse_answer_file(args.output_file)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
