import torch
import json
import csv
import random

class CNNDM_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split, tokenizer, max_encoder_length, max_decoder_length) -> None:
        super().__init__()
        self.split = split
        self.data = []

        for input, target in self.read_data(path, split):
            if split == 'train':
                self.make_input(tokenizer, input, target, max_encoder_length, max_decoder_length)
            else:
                self.data.append({
                    "inputs": input,
                    "targets": target
                })

    def shift_tokens_right(self, input_ids, pad_token_id: int=0, decoder_start_token_id: int=0):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def make_input(self, tokenizer, inputs, targets, max_encoder_length, max_decoder_length):
        model_inputs = tokenizer(inputs, max_length=max_encoder_length, padding="max_length", truncation=True)
        labels = tokenizer(targets, max_length=max_decoder_length, padding="max_length", truncation=True)

        labels["input_ids"] = torch.LongTensor([l if l != tokenizer.pad_token_id else -100 for l in labels["input_ids"]])

        model_inputs['input_ids'] = torch.LongTensor(model_inputs['input_ids'])
        model_inputs['attention_mask'] = torch.LongTensor(model_inputs['attention_mask'])
        model_inputs["decoder_input_ids"] = self.shift_tokens_right(labels["input_ids"])
        model_inputs["targets"] = labels["input_ids"]
        model_inputs["decoder_attention_mask"] = torch.LongTensor(labels["attention_mask"])

        self.data.append(model_inputs)

    def generate_input(self, question, context):
        return 

    def read_data(self, path, split):
        if split == 'dev': split = "validation"
        path = f"{path}/{split}.csv"
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                input = " ".join(["summarize:", row[1].lstrip()])
                if split=='train':
                    target = row[2]
                else:
                    target = row[2]
                yield input, target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            model_inputs = self.data[idx]
            for key, value in model_inputs.items():
                model_inputs[key] = value.cuda()
            return model_inputs
        else:
            return self.data[idx]