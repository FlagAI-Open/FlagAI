# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import psutil
import torch
from torch.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer
#from flagai.env_trainer import EnvTrainer
from flagai.env_trainer_v1 import EnvTrainer
from flagai.env_args import EnvArgs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can input all parameters by the command line.
# For example: python train_env_trainer.py --epochs=300 --batch_size=4 --env_type=pytorch
env_args = EnvArgs(
    env_type="bmtrain",
    experiment_name="gpt_13b",
    batch_size=4,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=2,
    log_interval=1,
    eval_interval=10000,
    num_gpus=1,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_gpt_13b",
    checkpoint_activations=False,
    save_interval=100000,
    fp16=True,
    training_script=__file__,
)
env_args = env_args.parse_args()

trainer = EnvTrainer(env_args)

'''
trainer = Trainer(
    env_type="bmtrain",
    experiment_name="gpt3_13b",
    batch_size=16,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=10,
    log_interval=10,
    eval_interval=10000,
    num_gpus=2,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_gpt3_13b",
    checkpoint_activations=False,
    save_interval=1000,
    fp16=True,
    hostfile='./hostfile',
    training_script=__file__
)
'''

## 
enable_debug = False
## 
if enable_debug:
    trainer.set_seed(2023)

## TODO
model_dir = "./"
os.makedirs(model_dir, exist_ok=True)
maxlen = 2048

from flagai.data.tokenizer import Tokenizer
model_name = "gpt3-13b-en"
cache_dir = model_dir + model_name
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print('*'*20, "tokenizer", tokenizer)

config_file = model_dir + model_name + "/config.json"
print('*'*20, "config_file", config_file)
from flagai.model.gpt2_model import GPT2Model
model = GPT2Model.init_from_json(config_file=config_file)
print('*'*20, "model", model)
mem = psutil.virtual_memory()
print('*'*20, 'before pre_train', mem.used / 1024.0 ** 3)
trainer.pre_train(model)
mem = psutil.virtual_memory()
print('*'*20, 'after pre_train', mem.used / 1024.0 ** 3)


# TODO
data_path = '/wangldai47/Downloads/Data'

def read_file():
    src = []
    tgt = []

    if enable_debug:
        part_file = './debug.txt'
        part_file = '/share/project/ldwang/data/pile/train/00.txt'
    #path = '/share/project/ldwang/data/pile/train/'
    path = '%s/train/' % data_path
    lines_count = 0
    # if True: # enable_debug
    for part_file in os.listdir(path):
        filename = path+part_file
        # filename = part_file # enable_debug
        print('*'*20, "filename", filename)
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                lines_count += 1
                src.append(line.strip('\n').lower())
                if lines_count%10000==1:
                    print('*'*20, 'lines_count', lines_count)
    return src, src

def read_file_dev():
    src = []
    tgt = []

    if enable_debug:
        part_file = '/share/project/ldwang/data/pile/train/00.txt'
        part_file = './dev.txt'
    else:
        part_file = '/share/project/ldwang/data/pile/val.txt'
        part_file = '%s/val.txt' % data_path
    if True:
        filename = part_file
        # print('*'*20, "filename", filename)
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                src.append(line.strip('\n').lower())
    return src, src

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
        input_ids = padding(input_ids, max_length)[:,:maxlen]

        data = {
            "input_ids": input_ids,
            "labels": input_ids
        }
        return data

sents_src, sents_tgt = read_file()
data_len = len(sents_tgt)
print('*'*20, 'data_len', data_len)
print('*'*20, 'after read_file', mem.used / 1024.0 ** 3)

train_src = sents_src
train_tgt = train_src

sents_src, sents_tgt = read_file_dev()
data_len = len(sents_tgt)
print('*'*20, 'data_len dev', data_len)
print('*'*20, 'after read_file_dev', mem.used / 1024.0 ** 3)

val_src = sents_src
val_tgt = sents_tgt

train_dataset = GPT2Seq2seqDataset(train_src,
                                   train_tgt,
                                   tokenizer=tokenizer,
                                   maxlen=maxlen)
val_dataset = GPT2Seq2seqDataset(val_src,
                                 val_tgt,
                                 tokenizer=tokenizer,
                                 maxlen=maxlen)

trainer.do_train(model,
              train_dataset=train_dataset,
              valid_dataset=val_dataset,
              collate_fn=GPT2Seq2seqDataset.collate_fn,
              optimizer=None
              )

'''
trainer.train(model,
              train_dataset=train_dataset,
              valid_dataset=val_dataset,
              collate_fn=GPT2Seq2seqDataset.collate_fn,
              optimizer=None
              )
'''

