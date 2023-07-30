# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import sys
sys.path.append('/share/project/liuguang/flagai-internal')
import torch
import os
from torch.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# single gpu
trainer = Trainer(
    env_type="deepspeed+mpu",
    experiment_name="roberta_seq2seq",
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=10,
    log_interval=10,
    eval_interval=10000,
    # load_dir='state_dicts',
    pytorch_device=device,
    save_dir="checkpoints",
    save_interval=1,
    num_checkpoints=1,
    save_optim=True,
    save_rng=True,
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=6,
    checkpoint_activations=True,
    model_parallel_size=1,
    hostfile='./hostfile',
    deepspeed_config='./deepspeed.json',
    training_script=__file__,
    load_type='latest'
)
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = cur_dir + '/data/train.src'
tgt_dir = cur_dir + '/data/train.tgt'
model_dir = "./state_dict/"
os.makedirs(model_dir, exist_ok=True)
maxlen = 1024
auto_loader = AutoLoader(
    "lm",
    model_name="aquila-7b",
    model_dir=model_dir,
    only_download_config=True,
    use_cache=False
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()


def read_file():
    src = []
    tgt = []

    with open(src_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            src.append(line.strip('\n').lower())

    with open(tgt_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tgt.append(line.strip('\n').lower())

    return src, tgt


class GPT2Seq2seqDataset(Dataset):

    def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=512):
        super(GPT2Seq2seqDataset, self).__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __getitem__(self, i):
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        data = self.tokenizer.encode_plus(src, tgt, max_length=self.maxlen)

        output = {
            "input_ids": data["input_ids"],
        }
        return output

    def __len__(self):

        return len(self.sents_src)

    @staticmethod
    def collate_fn(batch):
        def padding(indice, max_length, pad_idx=0):
            pad_indice = [
                item + [pad_idx] * max(0, max_length - len(item)) for item in indice
            ]
            return torch.tensor(pad_indice)

        input_ids = [data["input_ids"] for data in batch]
        max_length = max([len(t) for t in input_ids])
        input_ids = padding(input_ids, max_length)

        data = {
            "input_ids": input_ids,
            "labels": input_ids
        }
        return data

sents_src, sents_tgt = read_file()
data_len = len(sents_tgt)
train_size = int(data_len * 0.8)

train_src = sents_src[:train_size][:2000]
train_tgt = sents_tgt[:train_size][:2000]

val_src = sents_src[train_size:]
val_tgt = sents_tgt[train_size:]

train_dataset = GPT2Seq2seqDataset(train_src,
                                   train_tgt,
                                   tokenizer=tokenizer,
                                   maxlen=maxlen)
val_dataset = GPT2Seq2seqDataset(val_src,
                                 val_tgt,
                                 tokenizer=tokenizer,
                                 maxlen=maxlen)

# optimizer = torch.optim.Adam(model.parameters(),
#                              lr=1e-5,
#                              weight_decay=1e-5)
trainer.train(model,
              train_dataset=train_dataset,
            #   valid_dataset=val_dataset,
              collate_fn=GPT2Seq2seqDataset.collate_fn,
            #   optimizer=optimizer,
              )
