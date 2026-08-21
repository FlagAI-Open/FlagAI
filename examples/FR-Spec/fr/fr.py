from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import torch
import argparse
import os

def main(args):
	ds = load_dataset('cerebras/SlimPajama-627B', streaming=True, split='train')
	tokenizer = AutoTokenizer.from_pretrained(args.model_path)

	token_counter = Counter()
	num_lines = args.num_lines
	num_tokens = 0
	for i, d in tqdm(enumerate(ds)):
		tokens = tokenizer.encode(d['text'])
		token_counter.update(tokens)
		num_tokens += len(tokens)
		if i == num_lines:
			break

	sort_by_freq = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
	ids, frequencies = zip(*sort_by_freq)
	ids = list(ids)

	print(f"processed {num_lines} items")
	print(f"processed {num_tokens} tokens")

	if not os.path.exists(f'fr-index/{args.model_name}'):
			os.makedirs(f'fr-index/{args.model_name}')
			
	for r in args.vocab_size:
		eos_id = tokenizer.encode(tokenizer.special_tokens_map['eos_token'])
		if eos_id not in ids[:r]:
			not_in_ids = len(set(eos_id) - set(ids[:r]))
			freq_ids = ids[:r - not_in_ids] + eos_id
		else:
			freq_ids = ids[:r]
		
		print(f'save freq_{r}.pt, size:', len(freq_ids))
		with open(f'lmh_index/{args.model_name}/freq_{r}.pt', 'wb') as f:
			torch.save(freq_ids, f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--model_name', 
		type=str, 
		default='llama3-8b-instruct',
		help='The name of the model.'
	)
	parser.add_argument(
		'--model_path', 
		type=str, 
		default='meta-llama/Llama-3-8B-Instruct',
		help='The path to the model.'
	)
	parser.add_argument(
		'--num_lines', 
		type=int, 
		default=1000000, 
		help='The number of SlimPajama lines to process.'
	)
	parser.add_argument(
		'--vocab_size',
		nargs='+',
		type=int,
		default=[8192, 16384, 32768, 65536],
		help='The vocab sizes to process.'
	)
	
	args = parser.parse_args()
	print(args)
	main(args)
