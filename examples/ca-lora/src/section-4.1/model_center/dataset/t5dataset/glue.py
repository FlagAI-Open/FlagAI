import os
import torch
import transformers
from torch.utils.data import Dataset
from datasets import load_from_disk

class GLUE_Dataset(Dataset):
    def __init__(self, base_path, dataset_name, split, tokenizer, args):
        self.tokenizer = tokenizer
        self.max_length = args.max_encoder_length
        
        self.data_path = os.path.join(os.path.join(base_path, dataset_name), "data")

        if split == 'dev':
            split = 'validation_matched' if dataset_name=="mnli" else "validation"
            # mode = 'validation_mismatched'
        self.dataset = load_from_disk(self.data_path)[split]
        self.len = len(self.dataset)

    def __getitem__(self, index):
        return self.process(self.dataset[index])

    def __len__(self):
        return self.len


class MNLI_Dataset(GLUE_Dataset):
    def process(self, item):
        input = self.tokenizer(
            f"Sentence 1: {item['premise']} Sentence 2: {item['hypothesis']} Does sentence 1 entails sentence 2? <extra_id_0>.",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids' : input['input_ids'][0],
            'attention_mask' : input['attention_mask'][0],
            'labels' : torch.tensor(item['label']),
        }

    def get_verbalizer(self):
        return [self.tokenizer.encode("Yes")[0], self.tokenizer.encode("Maybe")[0], self.tokenizer.encode("No")[0]]


class QQP_Dataset(GLUE_Dataset):
    def process(self, item):
        input = self.tokenizer(
            f"Question 1: {item['question1']}\nQuestion 2: {item['question2']}\nAre the two questions paraphrase of each other? <extra_id_0>.",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids' : input['input_ids'][0],
            'attention_mask' : input['attention_mask'][0],
            'labels' : torch.tensor(item['label']),
        }
    
    def get_verbalizer(self):
        return [self.tokenizer.encode("No")[0], self.tokenizer.encode("Yes")[0]]


class QNLI_Dataset(GLUE_Dataset):
    def process(self, item):
        input = self.tokenizer(
            f"Question: {item['question']}\nSentence: {item['sentence']}\nDoes the sentence contains the answer to the question? <extra_id_0>.",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids' : input['input_ids'][0],
            'attention_mask' : input['attention_mask'][0],
            'labels' : torch.tensor(item['label']),
        }
    
    def get_verbalizer(self):
        return [self.tokenizer.encode("Yes")[0], self.tokenizer.encode("No")[0]]


class SST2_Dataset(GLUE_Dataset):
    def process(self, item):
        input = self.tokenizer(
            f"Sentence: {item['sentence']}\nDoes this sentence express positive or negative emotions? <extra_id_0>.",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids' : input['input_ids'][0],
            'attention_mask' : input['attention_mask'][0],
            'labels' : torch.tensor(item['label']),
        }
    
    def get_verbalizer(self):
        return [self.tokenizer.encode("negative")[0], self.tokenizer.encode("positive")[0]]

    def __getitem__(self, index):
        return self.process(self.dataset[index])

    def __len__(self):
        return self.len


class MRPC_Dataset(GLUE_Dataset):
    def process(self, item):
        input = self.tokenizer(
            f"Sentence 1: {item['sentence1']}\nSentence 2: {item['sentence2']}\nAre the two sentences paraphrase of each other? <extra_id_0>.",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids' : input['input_ids'][0],
            'attention_mask' : input['attention_mask'][0],
            'labels' : torch.tensor(item['label']),
        }
    
    def get_verbalizer(self):
        return [self.tokenizer.encode("No")[0], self.tokenizer.encode("Yes")[0]]