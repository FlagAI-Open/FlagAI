# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer
from flagai.data.collate_utils import seq2seq_collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cur_dir = os.path.dirname(os.path.abspath(__file__))
train_path = cur_dir + "/data/news.tsv"
# single gpu
trainer = Trainer(
    env_type="pytorch",
    experiment_name="roberta_seq2seq",
    batch_size=8,
    gradient_accumulation_steps=1,
    lr=1e-5,
    weight_decay=1e-3,
    epochs=10,
    log_interval=100,
    eval_interval=10000,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_seq2seq",
    checkpoint_activations=True,
    save_epoch=1,
)

model_dir = "../state_dict/"  # 模型位置

os.makedirs(model_dir, exist_ok=True)
maxlen = 256

auto_loader = AutoLoader(
    "seq2seq",
    model_name="BERT-base-en",
    model_dir=model_dir,
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()


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


class BertSeq2seqDataset(Dataset):

    def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=512):
        super(BertSeq2seqDataset, self).__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __getitem__(self, i):
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]

        data = self.tokenizer.encode_plus(src,
                                          tgt,
                                          max_length=self.maxlen,
                                          truncation=True)
        output = {
            "input_ids": data["input_ids"],
            "segment_ids": data["token_type_ids"],
        }
        return output

    def __len__(self):

        return len(self.sents_src)


sents_src, sents_tgt = read_file()

print(sents_src[10])
print(sents_tgt[10])

print(len(sents_src))

data_len = len(sents_tgt)
train_size = int(data_len * 0.9)

train_src = sents_src[:train_size]
train_tgt = sents_tgt[:train_size]

val_src = sents_src[train_size:]
val_tgt = sents_tgt[train_size:]

train_dataset = BertSeq2seqDataset(train_src,
                                   train_tgt,
                                   tokenizer=tokenizer,
                                   maxlen=maxlen)
val_dataset = BertSeq2seqDataset(val_src,
                                 val_tgt,
                                 tokenizer=tokenizer,
                                 maxlen=maxlen)

trainer.train(
    model,
    train_dataset=train_dataset,
    valid_dataset=val_dataset,
    collate_fn=seq2seq_collate_fn,
)
