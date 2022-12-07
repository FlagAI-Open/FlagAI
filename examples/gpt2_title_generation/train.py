# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# single gpu
trainer = Trainer(
    env_type="bmtrain",
    experiment_name="gpt2_title-generation",
    batch_size=8,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=10,
    log_interval=10,
    eval_interval=10000,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_gpt2_title-generation",
    checkpoint_activations=True,
    save_interval=10000,
    hostfile='./hostfile'
)

cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = cur_dir + '/data/train.src'
tgt_dir = cur_dir + '/data/train.tgt'
model_dir = "./state_dict/"
os.makedirs(model_dir, exist_ok=True)
maxlen = 256

auto_loader = AutoLoader(
    "seq2seq",
    model_name="GPT2-base-ch",
    model_dir=model_dir,
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

optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-5,
                             weight_decay=1e-5)

trainer.train(model,
              train_dataset=train_dataset,
              valid_dataset=val_dataset,
              collate_fn=GPT2Seq2seqDataset.collate_fn,
              optimizer=optimizer
              )
