# Copyright © 2022 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License")
import sys
import os
import torch
from torch.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(
    env_type="deepspeed+mpu",
    experiment_name="llama_seq2seq",
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=10,
    fp16=True,
    log_interval=1,
    eval_interval=10000,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints",
    save_interval=1000,
    num_checkpoints=1,
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=2,
    checkpoint_activations=False,
    model_parallel_size=2,
    hostfile='./hostfile',
    deepspeed_config='./deepspeed.json',
    training_script=__file__,
)
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = cur_dir + '/data/train.src'
tgt_dir = cur_dir + '/data/train.tgt'
maxlen = 256

auto_loader = AutoLoader(
    "lm",
    model_name="llama-13b-en",
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

    def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=21):
        super(GPT2Seq2seqDataset, self).__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __getitem__(self, i):
        src = self.sents_src[i][:512]
        tgt = self.sents_tgt[i]
        in_text = f"{src}。对以上文字提取重点:{tgt}"
        
        data = self.tokenizer.encode(in_text, bos=True, eos=True)

        output = {
            "input_ids": data,
        }
        return output

    def __len__(self):

        return len(self.sents_src)


sents_src, sents_tgt = read_file()
data_len = len(sents_tgt)
train_size = int(data_len * 0.8)

train_src = sents_src[:train_size]
train_tgt = sents_tgt[:train_size]

train_dataset = GPT2Seq2seqDataset(train_src,
                                   train_tgt,
                                   tokenizer=tokenizer,
                                   maxlen=maxlen)

# optimizer = torch.optim.Adam(model.parameters(),
#                              lr=1e-5,
#                              weight_decay=1e-5)
trainer.train(model,
              train_dataset=train_dataset,
              collate_fn=GPT2Seq2seqDataset.collate_fn,
            #   optimizer=optimizer,
              )
