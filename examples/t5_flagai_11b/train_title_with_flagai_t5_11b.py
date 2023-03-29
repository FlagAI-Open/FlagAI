# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.trainer import Trainer
from flagai.model.t5_model import T5ForConditionalGeneration
from transformers import T5Tokenizer
from flagai.data.tokenizer import Tokenizer
from flagai.model.predictor.predictor import Predictor
from torch.utils.data import Dataset
import os
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))

train_path = "../bert_title_generation_english/data/news.tsv"

trainer = Trainer(env_type='deepspeed',
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

# t5-11b is not uploaded to modelhub yet. Since it shares tokenizer with T5-base-en, we will get tokenizer here
tokenizer = Tokenizer.from_pretrained('T5-base-en')
 # path to your downloaded model files is /mnt/t5-11b
model = T5ForConditionalGeneration.from_pretrain(download_path='/mnt',
                                                 model_name='t5-11b',checkpoint_activations=True)

print("loading model & tokenizer is done!")

maxlen = 1024

predictor = Predictor(model, tokenizer)

class T5Seq2seqDataset(Dataset):

    def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=512):
        super(T5Seq2seqDataset, self).__init__()
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
        output['target_ids'] = labels.input_ids
        return output

    def __len__(self):
        return len(self.sents_src)

def t5_seq2seq_collate_fn(batch):

    def padding(indice, max_length, pad_idx=0):

        pad_indice = [
            item + [pad_idx] * max(0, max_length - len(item))
            for item in indice
        ]
        return torch.tensor(pad_indice)

    token_ids_src = [data["input_ids"] for data in batch]
    max_length_src = max([len(t) for t in token_ids_src])
    token_ids_tgt = [data["target_ids"] for data in batch]
    max_length_tgt = max([len(t) for t in token_ids_tgt])

    token_ids_padded = padding(token_ids_src, max_length_src)
    target_ids_padded = padding(token_ids_tgt, max_length_tgt)
    labels_ids = target_ids_padded.clone()
    labels_ids[labels_ids == 0] = -100
    target_ids_padded = target_ids_padded[:, :-1].contiguous()
    labels_ids = labels_ids[:, 1:].contiguous()

    return {
        "input_ids": token_ids_padded,
        "decoder_input_ids": target_ids_padded,
        "labels": labels_ids
    }

train_src, train_tgt = read_file()

train_dataset = T5Seq2seqDataset(train_src,
                                 train_tgt,
                                 tokenizer=tokenizer,
                                 maxlen=maxlen)

trainer.train(model,
              train_dataset=train_dataset,
              collate_fn=t5_seq2seq_collate_fn)
