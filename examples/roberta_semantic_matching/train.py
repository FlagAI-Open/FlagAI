# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer
from flagai.data.collate_utils import bert_cls_collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target = [0, 1]
trainer = Trainer(env_type="pytorch",
                  experiment_name="roberta-base-ch-semantic-matching",
                  batch_size=8,
                  gradient_accumulation_steps=1,
                  lr=1e-5,
                  weight_decay=1e-3,
                  epochs=10,
                  log_interval=100,
                  eval_interval=500,
                  load_dir=None,
                  pytorch_device=device,
                  save_dir="checkpoints_semantic_matching",
                  save_interval=1)

cur_dir = os.path.dirname(os.path.abspath(__file__))
train_path = cur_dir + "/data/train.tsv"
model_dir = "./checkpoints/"
maxlen = 256

auto_loader = AutoLoader("semantic-matching",
                         model_name="RoBERTa-base-ch",
                         model_dir=model_dir,
                         class_num=len(target))
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()


def read_corpus(data_path):
    sents_src = []
    sents_tgt = []

    with open(data_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split("\t")
        if len(line) == 3:
            sents_tgt.append(int(line[2]))
            sents_src.append([line[0], line[1]])

    return sents_src, sents_tgt


class BertSemanticMatchDataset(Dataset):

    def __init__(self, sents_src, sents_tgt):
        super(BertSemanticMatchDataset, self).__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

    def __getitem__(self, i):
        src_1, src_2 = self.sents_src[i][0], self.sents_src[i][1]
        tgt = self.sents_tgt[i]
        data = tokenizer.encode_plus(src_1,
                                     src_2,
                                     max_length=maxlen,
                                     truncation=True)

        output = {
            "input_ids": data["input_ids"],
            "segment_ids": data["token_type_ids"],
            "labels": tgt
        }
        return output

    def __len__(self):

        return len(self.sents_src)


def eval_metric(logits, labels, **kwargs):
    logits = logits.argmax(dim=-1)
    return (logits == labels).sum()


def main():
    src, tgt = read_corpus(data_path=train_path)
    data_len = len(src)
    train_size = int(data_len * 0.9)
    train_src = src[:train_size]
    train_tgt = tgt[:train_size]

    val_src = src[train_size:]
    val_tgt = tgt[train_size:]

    train_dataset = BertSemanticMatchDataset(train_src, train_tgt)
    val_dataset = BertSemanticMatchDataset(val_src, val_tgt)

    trainer.train(model,
                  train_dataset=train_dataset,
                  valid_dataset=val_dataset,
                  collate_fn=bert_cls_collate_fn,
                  metric_methods=[["acc", eval_metric]])


if __name__ == '__main__':
    main()
