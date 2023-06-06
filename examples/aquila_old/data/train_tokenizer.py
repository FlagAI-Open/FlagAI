

import glob
import jsonlines

def get_training_corpus(batch_size=32):
    files = []
    for name in open('./files.txt'):
        files.append(name.strip())
    print(files)
    samples = []
    for f_jsonl in files:
        f = jsonlines.open(f_jsonl)
        for line in f:
            if len(samples) == batch_size:
                yield samples
                samples = []
            sample = line["text"]
            samples.append(sample)
    if len(samples) == batch_size:
        yield samples

import transformers
from transformers import AutoTokenizer
old_tokenizer_dir = 'gpt2'
old_tokenizer = AutoTokenizer.from_pretrained(old_tokenizer_dir)

#old_tokenizer_dir = 'decapoda-research/llama-7b-hf'
#old_tokenizer = transformers.LlamaTokenizer.from_pretrained(old_tokenizer_dir)

vocab_size = 130000
vocab_size = 100000
vocab_size = 80000
training_corpus = get_training_corpus()
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size)

tokenizer_dir = 'gpt2_new_v2'
tokenizer_dir = 'gpt2_new_1w'
tokenizer_dir = 'gpt2_new_80k'
tokenizer.save_pretrained(tokenizer_dir)


