# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset
import gc
gc.collect()
torch.cuda.empty_cache()

from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
from flagai.env_args import EnvArgs
from flagai.env_trainer_v1 import EnvTrainer

#torch.autograd.set_detect_anomaly(True)

from examples.aquila.build_index_mappings import _build_train_valid_test_datasets
from examples.aquila.build_index_mappings import _build_train_valid_test_weighted_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can input all parameters by the command line.
# For example: python train_env_trainer.py --epochs=300 --batch_size=4 --env_type=pytorch
env_args = EnvArgs(
    env_type="bmtrain",
    experiment_name="aquila",
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=100,
    log_interval=10,
    eval_interval=5000,
    num_gpus=1,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_aquila",
    checkpoint_activations=False,
    save_interval=5000,
    fp16=True,
    training_script=__file__,
)
env_args = env_args.parse_args()
#env_args.wandb = False

# overwrite
if env_args.yaml_config:
    import yaml
    file_data = open(env_args.yaml_config, 'r', encoding="utf-8").read()
    data = yaml.load_all(file_data)
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

print(f"Trainer effective env_args={env_args} local_rank={trainer.local_rank}", flush=True)

checkpoints = env_args.pre_load_dir

model_name = env_args.model_name

env_args.enable_sft_conversations_dataset_v3 = True


print('*'*20, "model_name", model_name, flush=True)

'''
auto_loader = AutoLoader(
    "lm",
    model_name=model_name,
    model_dir=checkpoints,
    only_download_config=True,
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
print('*'*20, "model", model)
trainer.pre_train(model)
print('*'*20, "model", model)

'''

cache_dir = os.path.join(checkpoints, model_name)
print('*'*20, "cache_dir", cache_dir)
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print('*'*20, "tokenizer", tokenizer)

# avoid sync loading models in case of Mem OOM
if env_args.bmt_async_load:
    import time
    time.sleep(10*60*(trainer.local_rank%4))


config_file = os.path.join(cache_dir, 'config.json')
from flagai.model.aquila_model import AQUILAModel
model = AQUILAModel.init_from_json(config_file=config_file)
print('*'*20, "model", model)

## bmt_pre_load
checkpoint_path = os.path.join(cache_dir, "pytorch_model.bin")
if env_args.bmt_pre_load:
    model.load_weights(checkpoint_path)

trainer.pre_train(model)

print('*'*20, "model", model, flush=True)


## Use Prebuilt DataSets
data_prefix = '/data/yzd/FlagAI/examples/indexed_dataset/data/demo_text_document'
data_impl = 'mmap'
splits_string = '90,10'
train_valid_test_num_samples = [90, 10]
seq_length = 1024
seed = 2023
skip_warmup = True

train_dataset, valid_dataset, _ = _build_train_valid_test_datasets(
    data_prefix, data_impl, splits_string,
    train_valid_test_num_samples,
    seq_length, seed, skip_warmup)
print("Total train_dataset: ", len(train_dataset), flush=True)
print("Total valid_dataset: ", len(valid_dataset), flush=True)

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
    valid_dataset=None,
    collate_fn=collate_fn,
    optimizer=None,
    rank_split=False)
