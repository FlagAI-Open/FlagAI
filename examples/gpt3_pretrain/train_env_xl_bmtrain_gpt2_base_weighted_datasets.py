# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer
#from flagai.env_trainer import EnvTrainer
from flagai.env_trainer_v1 import EnvTrainer
from flagai.env_args import EnvArgs
from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_datasets
from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_weighted_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can input all parameters by the command line.
# For example: python train_env_trainer.py --epochs=300 --batch_size=4 --env_type=pytorch
env_args = EnvArgs(
    env_type="bmtrain",
    experiment_name="gpt2_base",
    batch_size=16,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=1,
    log_interval=1,
    eval_interval=10000,
    num_gpus=2,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_gpt2_base",
    checkpoint_activations=False,
    save_interval=10,
    fp16=True,
    training_script=__file__,
)
env_args = env_args.parse_args()

'''
env_args.save_rng = True
env_args.save_optim = True

env_args.load_dir = 'checkpoints_gpt2_base.bak'
env_args.load_rng = True
env_args.load_optim = True
'''

trainer = EnvTrainer(env_args)

# Trainer as Trigger
if not env_args.not_call_launch:
    import sys
    sys.exit(0)

## TODO
model_dir = "./"
os.makedirs(model_dir, exist_ok=True)

from flagai.data.tokenizer import Tokenizer
model_name = "gpt2-base-en"
cache_dir = model_dir + model_name
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# print('*'*20, "tokenizer", tokenizer)

config_file = model_dir + model_name + "/config.json"
# print('*'*20, "config_file", config_file)
from flagai.model.gpt2_model import GPT2Model
model = GPT2Model.init_from_json(config_file=config_file)
print('*'*20, "model", model)
trainer.pre_train(model)

## Use Weighted Datasets
## weight01, prefix01, weight02, prefix02
data_prefix = [
    200,
    '/share/project/ldwang/data/indexed_dataset/merged/OpenWebText2_merged_text_document',
    100,
    '/share/project/ldwang/data/indexed_dataset/merged/dedup_wudao_5pct_merged_text_document',
]
data_impl = 'mmap'
## splits_string len should same as train_valid_test_num_samples len
splits_string = '9999,1'
## rebuilding if no npy files for train_valid_test_num_samples config
## suggested than preprocess before run
train_valid_test_num_samples = [26944801, 2695]
seq_length = 1024
seed = 2023
skip_warmup = True

train_dataset, val_dataset, test_dataset = _build_train_valid_test_weighted_datasets(
    data_prefix, data_impl, splits_string,
    train_valid_test_num_samples,
    seq_length, seed, skip_warmup)

def collate_fn(batch):
    def padding(indice, max_length, pad_idx=tokenizer.token_end_id):
        pad_indice = [
            item.tolist() + [pad_idx] * max(0, max_length - len(item.tolist())) for item in indice
        ]
        return torch.tensor(pad_indice)

    input_ids = [data["input_ids"] for data in batch]
    max_length = max([len(t) for t in input_ids])
    input_ids = padding(input_ids, max_length)[:,:seq_length]

    data = {
        "input_ids": input_ids,
        "labels": input_ids
    }
    return data

trainer.do_train(
    train_dataset=train_dataset,
    valid_dataset=val_dataset,
    collate_fn=collate_fn,
    optimizer=None,
    rank_split=False)

