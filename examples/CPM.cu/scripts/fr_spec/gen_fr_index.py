from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import torch
import argparse
import os

def main(args):
	print(f"Generating FR index for {args.model_name} with vocab size {args.vocab_size}")
	print(f"This may take about 5-10 minutes with 1M lines on good network connection")
	print("Loading dataset...")
	ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', streaming=True)['train']
	# Only take the number of samples we need to process
	ds = ds.take(args.num_lines + 1)  # +1 to account for 0-indexing
	print(f"Dataset limited to {args.num_lines + 1} samples")
	
	print("Loading tokenizer...")
	tokenizer = AutoTokenizer.from_pretrained(args.model_path)
	print("Tokenizer loaded successfully")

	token_counter = Counter()
	num_lines = args.num_lines
	num_tokens = 0
	print("Starting to process data...")
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
	print(f"unique tokens: {len(ids)}")
	
	os.makedirs(f'fr_index/{args.model_name}', exist_ok=True)
			
	for r in args.vocab_size:
		eos_id = tokenizer.encode(tokenizer.special_tokens_map['eos_token'])
		if eos_id not in ids[:r]:
			not_in_ids = len(set(eos_id) - set(ids[:r]))
			freq_ids = ids[:r - not_in_ids] + eos_id
		else:
			freq_ids = ids[:r]
		if (r != len(freq_ids)):
			print(f"Warning: requested vocab_size {r} but actual size: {len(freq_ids)}, file not saved")
		else:
			pt_path = f'fr_index/{args.model_name}/freq_{r}.pt'
			print(f'save {pt_path}, actual size: {len(freq_ids)}')
			with open(pt_path, 'wb') as f:
				torch.save(freq_ids, f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--model_name', 
		type=str, 
		default='MiniCPM4-8B',
		help='The name of the model.'
	)
	parser.add_argument(
		'--model_path', 
		type=str, 
		default='openbmb/MiniCPM4-8B',
		help='The path to the model.'
	)
	parser.add_argument(
		'--num_lines', 
		type=int, 
		default=1000000, 
		help='The number of lines to process.'
	)
	parser.add_argument(
		'--vocab_size',
		nargs='+',
		type=int,
		default=[8192, 16384, 32768],
		help='The vocab sizes to process.'
	)
	
	args = parser.parse_args()
	print(args)
	main(args)
