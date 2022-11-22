# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os

import torch
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

from flagai.trainer import Trainer


class MyTrainer(Trainer):

    def forward_step(self, data, model, mems):

        model_outputs = model(**data)
        output = {}
        output['loss'] = model_outputs.loss
        output['logits'] = model_outputs.logits
        output['hidden_states'] = model_outputs.decoder_hidden_states
        return output


trainer = MyTrainer(env_type='deepspeed',
                    epochs=1,
                    batch_size=1,
                    eval_interval=100000,
                    log_interval=1,
                    experiment_name='t5-11b',
                    load_dir=None,
                    lr=1e-4,
                    fp16=True,
                    master_ip='127.0.0.1',
                    master_port=17755,
                    num_nodes=1,
                    num_gpus=1,
                    hostfile='./hostfile',
                    model_parallel_size=1,
                    deepspeed_config='./deepspeed.json',
                    training_script=__file__)

model_name = 't5-11b'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained('/mnt/t5-11b')
model.gradient_checkpointing = True

print("loading model & tokenizer is done!")
train_path = "../bert_title_generation_english/data/news.tsv"

maxlen = 1024

def read_file():
    src = []
    tgt = []

    index = 0
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            index += 1
            if index == 1:
                continue
            line = line.strip('\n').split('\t')
            src_list = line[4].split(" ")
            if len(src_list) > 510:
                continue

            src.append(line[4])
            tgt.append(line[3])
            if index == 100000:
                break

    return src, tgt

class T5Seq2seqDataset(Dataset):

    def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=512):
        super().__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __getitem__(self, i):
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        inputs = tokenizer(src)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(tgt)
        output = {}
        output['input_ids'] = inputs.input_ids
        output['labels'] = labels.input_ids
        return output

    def __len__(self):
        return len(self.sents_src)


def seq2seq_collate_fn(batch):

    def padding(indice, max_length, pad_idx=0):

        pad_indice = [
            item + [pad_idx] * max(0, max_length - len(item))
            for item in indice
        ]
        return torch.tensor(pad_indice)

    token_ids = [data["input_ids"] for data in batch]
    max_length_tk = max([len(t) for t in token_ids])
    labels = [data["labels"] for data in batch]
    max_length_lb = max([len(t) for t in labels])

    token_ids_padded = padding(token_ids, max_length_tk)
    labels_padded = padding(labels, max_length_lb)

    data = {"input_ids": token_ids_padded, "labels": labels_padded}

    return data


sents_src, sents_tgt = read_file()
data_len = len(sents_tgt)
train_size = int(data_len * 0.8)
train_src = sents_src[:train_size][:200]
train_tgt = sents_tgt[:train_size][:200]

val_src = sents_src[train_size:]
val_tgt = sents_tgt[train_size:]

train_dataset = T5Seq2seqDataset(train_src,
                                 train_tgt,
                                 tokenizer=tokenizer,
                                 maxlen=maxlen)
val_dataset = T5Seq2seqDataset(val_src,
                               val_tgt,
                               tokenizer=tokenizer,
                               maxlen=maxlen)

trainer.train(model,
              train_dataset=train_dataset,
              collate_fn=seq2seq_collate_fn)
