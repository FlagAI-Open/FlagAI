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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can input all parameters by the command line.
# For example: python train_env_trainer.py --epochs=300 --batch_size=4 --env_type=pytorch
env_args = EnvArgs(
    env_type="bmtrain",
    experiment_name="galactica_6.7b",
    batch_size=16,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=1,
    log_interval=50,
    eval_interval=10000,
    num_gpus=1,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_galactica_6.7b",
    checkpoint_activations=False,
    save_interval=10000,
    fp16=True,
    training_script=__file__,
)
env_args = env_args.parse_args()

trainer = EnvTrainer(env_args)

# Trainer as Trigger
if not env_args.not_call_launch:
    import sys
    sys.exit(0)

## TODO
### 需要根据配置情况填写路径
model_dir = "./"
model_name = "galactica-6.7b"
auto_loader = AutoLoader(
    "lm",
    model_name=model_name,
    model_dir=model_dir,
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()

trainer.pre_train(model)

# TODO
### 需要根据数据集情况填写路径前缀
data_prefix = '/share/project/ldwang/data/indexed_dataset/gpm/merged_text_document'
data_impl = 'mmap'
splits_string = '9999,1,0'
train_valid_test_num_samples = [344379254, 34441, 0]
seq_length = 2048
seed = 2023
skip_warmup = True

train_dataset, val_dataset, test_dataset = _build_train_valid_test_datasets(
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

