# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(
    env_type="deepspeed",
    experiment_name="GLM_seq2seq",
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-5,
    epochs=10,
    num_gpus = 2,
    log_interval=10,
    eval_interval=40,
    load_dir=None,
    pytorch_device=device,
    # save_dir="checkpoints_glm_title_generation",
    save_interval=40,
    num_checkpoints=1,
    hostfile='./hostfile',
    deepspeed_config='./deepspeed.json'
)

# cur_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = cur_dir + '/data/train.src'
# tgt_dir = cur_dir + '/data/train.tgt'

src_dir = "./data/train.src"
tgt_dir = "./data/train.tgt"


maxlen = 256
auto_loader = AutoLoader("lm",
                         model_name="GLM-large-ch",
                         model_dir="./state_dict/")
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


class GLMSeq2seqDataset(Dataset):

    def __init__(self,
                 sents_src,
                 sents_tgt,
                 tokenizer,
                 max_src_length=300,
                 max_tgt_length=200):
        super(GLMSeq2seqDataset, self).__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.no_block_position = False

    def __getitem__(self, i):
        source_text = self.sents_src[i]
        target_text = self.sents_tgt[i]
        data = self.tokenizer.encode_plus(source_text=source_text, target_text=target_text)

        return data

    def __len__(self):

        return len(self.sents_src)


class GLMPoetryDynamicCollateFN():  #padding process in each batch

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad_token(self, tokens, max_length):
        pad_len = max_length - len(tokens)
        tokens += [self.pad_id] * pad_len
        return tokens

    def pad_position_ids(self, position_ids, max_length):
        pad_len = max_length - len(position_ids[0])
        position_ids[0] += [len(position_ids[0]) + x for x in range(pad_len)]
        position_ids[1] += [1] * pad_len
        return position_ids

    def pad_loss_mask(self, loss_mask, max_length):
        pad_len = max_length - len(loss_mask)
        loss_mask += [0] * pad_len
        return loss_mask

    def __call__(self, batch):
        input_ids = [data["input_ids"] for data in batch]
        target_ids = [data["target_ids"] for data in batch]
        position_ids = [data["position_ids"] for data in batch]
        attention_mask = [data['attention_mask'] for data in batch]
        loss_mask = [data['loss_mask'] for data in batch]

        max_length = max([len(t) for t in input_ids])
        for i in range(len(input_ids)):
            input_ids[i] = self.pad_token(input_ids[i], max_length)
            target_ids[i] = self.pad_token(target_ids[i], max_length)
            position_ids[i] = self.pad_position_ids(position_ids[i],
                                                    max_length)
            loss_mask[i] = self.pad_loss_mask(loss_mask[i], max_length)
        return {
            'input_ids': torch.LongTensor(input_ids),
            'labels': torch.LongTensor(target_ids),
            'position_ids': torch.LongTensor(position_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'loss_mask': torch.LongTensor(loss_mask)
        }


sents_src, sents_tgt = read_file()
my_collate_fn = GLMPoetryDynamicCollateFN(
    pad_id=tokenizer.get_command_id('pad'))

data_len = len(sents_tgt)
train_size = int(data_len * 0.8)
train_src = sents_src[:train_size][:2000]
train_tgt = sents_tgt[:train_size][:2000]

val_src = sents_src[train_size:]
val_tgt = sents_tgt[train_size:]

train_dataset = GLMSeq2seqDataset(train_src,
                                  train_tgt,
                                  tokenizer=tokenizer,
                                  max_src_length=300,
                                  max_tgt_length=200)
val_dataset = GLMSeq2seqDataset(val_src,
                                val_tgt,
                                tokenizer=tokenizer,
                                max_src_length=300,
                                max_tgt_length=200)

trainer.train(model,
              train_dataset=train_dataset,
              valid_dataset=val_dataset,
              collate_fn=my_collate_fn)
