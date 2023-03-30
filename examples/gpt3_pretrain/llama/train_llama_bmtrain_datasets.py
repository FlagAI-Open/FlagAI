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

from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_datasets
from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_weighted_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can input all parameters by the command line.
# For example: python train_env_trainer.py --epochs=300 --batch_size=4 --env_type=pytorch
env_args = EnvArgs(
    env_type="bmtrain",
    experiment_name="llama",
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
    save_dir="checkpoints_llama",
    checkpoint_activations=False,
    save_interval=5000,
    fp16=True,
    training_script=__file__,
)
env_args = env_args.parse_args()
#env_args.wandb = False

trainer = EnvTrainer(env_args)

# Trainer as Trigger
if not env_args.not_call_launch:
    import sys
    sys.exit(0)

## TODO
checkpoints = "/share/project/ldwang/checkpoints/"
checkpoints = "/data/ldwang/checkpoints/"
checkpoints = "/data/ldwang/state_dict/"
model_name = "llama-30b-en"
model_name = "llama-7b-en"

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
cache_dir = checkpoints + model_name
#print('*'*20, "cache_dir", cache_dir)
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#print('*'*20, "tokenizer", tokenizer)

config_file = cache_dir + "/config.json"
from flagai.model.llama_model import LLAMAModel
model = LLAMAModel.init_from_json(config_file=config_file)
#print('*'*20, "model", model)

trainer.pre_train(model)
print('*'*20, "model", model)

data_prefix = [
    2.7,
    '/data/indexed_dataset/batch1_tok100k/cn_baike_text_document',
    2.91,
    '/data/indexed_dataset/batch1_tok100k/cn_ebook_merge_maxlen_text_document',
    1.89,
    '/data/indexed_dataset/batch1_tok100k/cn_zhihu_text_document',
    1.46,
    '/data/indexed_dataset/batch1_tok100k/cn_wudao_base_text_document',
    1.01,
    '/data/indexed_dataset/batch1_tok100k/cn_wudao_dedup_merged_text_document',
    0.9,
    '/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-arxiv_text_document',
    2.5,
    '/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-bookcorpus2_text_document',
    1.1,
    '/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-books3_text_document',
    1.38,
    '/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-gutenberg_pg-19_text_document',
    2.82,
    '/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-openwebtext2_text_document',
    1.01,
    '/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-pile-cc_text_document',
    0.95,
    '/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-pubmed_abstracts_text_document',
    0.95,
    '/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-pubmed_central_text_document',
    2.08,
    '/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-stackexchange_text_document',
    1.46,
    '/data/indexed_dataset/batch1_tok100k/en_dedup-md5-pile-wikipedia_en_text_document',
]

data_impl = 'mmap'
## splits_string len should same as train_valid_test_num_samples len
splits_string = '9999,1'
## rebuilding if no npy files for train_valid_test_num_samples config
## 400B
train_valid_test_num_samples = [390585937, 39063]
seq_length = 2048
seed = 2023
skip_warmup = True

train_dataset, valid_dataset, _ = _build_train_valid_test_weighted_datasets(
    data_prefix, data_impl, splits_string,
    train_valid_test_num_samples,
    seq_length, seed, skip_warmup)
print("Total train_dataset: ", len(train_dataset))
print("Total valid_dataset: ", len(valid_dataset))

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
    valid_dataset=valid_dataset,
    collate_fn=collate_fn,
    optimizer=None,
    rank_split=False)

