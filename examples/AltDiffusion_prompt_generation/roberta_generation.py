#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/12/25 19:16
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

maxlen = 256


auto_loader = AutoLoader(
    "title-generation",
    model_name="RoBERTa-base-ch",
    model_dir=model_dir,
)
tokenizer = auto_loader.get_tokenizer()

# Data loading

import pandas as pd
train_data = pd.read_csv("data/train.csv", sep=',').values.tolist()
train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = pd.read_csv("data/dev.csv", sep=',').values.tolist()
eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

from torch.utils.data import DataLoader, Dataset

class BertTitleGenerationDataset(Dataset):
    def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=512):
        super(BertTitleGenerationDataset, self).__init__()
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
            "segment_ids": data["token_type_ids"],
        }
        return output

    def __len__(self):

        return len(self.sents_src)



train_src = train_df['input_text']
train_tgt = train_df['target_text']

val_src = eval_df['input_text']
val_tgt = eval_df['target_text']


train_dataset = BertTitleGenerationDataset(train_src,
                        train_tgt,
                        tokenizer=tokenizer,
                        maxlen=maxlen)
val_dataset = BertTitleGenerationDataset(val_src,
                      val_tgt,
                      tokenizer=tokenizer,
                      maxlen=maxlen)

# train
from flagai.trainer import Trainer
from flagai.data.collate_utils import seq2seq_collate_fn
trainer = Trainer(
    env_type="pytorch",
    experiment_name="RoBERTa-base-ch",
    batch_size=2,
    gradient_accumulation_steps=1,
    lr=1e-5,
    weight_decay=1e-3,
    epochs=100,
    log_interval=1,
    eval_interval=10,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints-bert-title-generation-en",
    checkpoint_activations=True,
    save_interval=1000,
    fp16 = False)
trainer.train(
    model,
    train_dataset=train_dataset,
    valid_dataset=val_dataset,
    collate_fn=seq2seq_collate_fn,
)

# test
predictor = Predictor(model, tokenizer)
test_data = [
    "裙,线条,百褶"
    ]
for text in test_data:
    print(
        predictor.predict_generate_beamsearch(text,
                                              out_max_length=50,
                                              beam_size=3))

