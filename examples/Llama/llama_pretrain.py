# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset
import gc
import json

gc.collect()
torch.cuda.empty_cache()

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from llama_bmt_monkey_patch import (
    replace_llama_attn_with_bmt,
)

from flagai.env_args import EnvArgs
from flagai.env_trainer_v1 import EnvTrainer
from flagai.data.dataset.indexed_dataset.build_index_mappings import _build_train_valid_test_datasets, _build_train_valid_test_weighted_datasets
import bmtrain as bmt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can input all parameters by the command line.
# For example: python train_env_trainer.py --epochs=300 --batch_size=4 --env_type=pytorch
env_args = EnvArgs(
    env_type="bmtrain",
    experiment_name="llama3",
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=10000,
    log_interval=1,
    eval_interval=5000,
    num_gpus=1,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_out",
    checkpoint_activations=False,
    save_interval=100,
    fp16=True,
    training_script=__file__,
)
env_args = env_args.parse_args()
#env_args.wandb = False

# overwrite
if env_args.yaml_config:
    import yaml
    file_data = open(env_args.yaml_config, 'r', encoding="utf-8").read()
    data = yaml.load_all(file_data, Loader=yaml.SafeLoader)
    delattr(env_args, 'yaml_config')
    arg_dict = env_args.__dict__
    for subdata in data:
        for key, value in subdata.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
trainer = EnvTrainer(env_args)

# Trainer as Trigger
if not env_args.not_call_launch:
    import sys
    sys.exit(0)

print(f"Trainer effective env_args={env_args} local_rank={os.environ['LOCAL_RANK']}",
      flush=True)
checkpoints = env_args.pre_load_dir
model_name = env_args.model_name

print('*' * 20, "model_name", model_name, flush=True)

cache_dir = os.path.join(checkpoints, model_name)
print('*' * 20, "cache_dir", cache_dir)
tokenizer = AutoTokenizer.from_pretrained(cache_dir)
print('*' * 20, "tokenizer", tokenizer)

# avoid sync loading models in case of Mem OOM
if env_args.bmt_async_load:
    import time
    time.sleep(10 * 60 * (os.environ['LOCAL_RANK'] % 4))

config_file = os.path.join(cache_dir, 'config.json')
with open(config_file, 'r') as f:
  model_args = json.load(f)

# bmt
replace_llama_attn_with_bmt()

model = LlamaForCausalLM.from_pretrained(cache_dir)

## bmt_pre_load

trainer.pre_train(model)

print('*' * 20, "model", model, flush=True)

## Use Prebuilt DataSets
data_prefix = '../indexed_dataset/data/demo_text_document'
data_impl = 'mmap'
splits_string = '90,10'
train_valid_test_num_samples = [90, 10]
seq_length = 1024
seed = 2023
skip_warmup = True

train_dataset, valid_dataset, _ = _build_train_valid_test_datasets(
    data_prefix, data_impl, splits_string, train_valid_test_num_samples,
    seq_length, seed, skip_warmup)
print("Total train_dataset: ", len(train_dataset), flush=True)
print("Total valid_dataset: ", len(valid_dataset), flush=True)


def collate_fn(batch):

    def padding(indice, max_length, pad_idx=0):
        pad_indice = [
            item.tolist() + [pad_idx] * max(0, max_length - len(item.tolist()))
            for item in indice
        ]
        return torch.tensor(pad_indice)

    input_ids = [data["input_ids"] for data in batch]
    max_length = max([len(t) for t in input_ids])
    input_ids = padding(input_ids, max_length)[:, :seq_length]

    data = {"input_ids": input_ids, "labels": input_ids}
    return data


trainer.do_train(train_dataset=train_dataset,
                 valid_dataset=None,
                 collate_fn=collate_fn,
                 optimizer=None,
                 rank_split=False)

