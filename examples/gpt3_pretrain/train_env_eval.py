# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import math
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
    experiment_name="gpt2_base",
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=1,
    log_interval=10,
    eval_interval=10000,
    num_gpus=4,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_gpt2_base",
    checkpoint_activations=False,
    save_interval=10000,
    fp16=False,
    training_script=__file__,
)
env_args = env_args.parse_args()
env_args.wandb = False

trainer = EnvTrainer(env_args)

# Trainer as Trigger
if not env_args.not_call_launch:
    import sys
    sys.exit(0)

model_dir = "./"
auto_loader = AutoLoader(
    "seq2seq",
    model_name="gpt2-base-en",
    model_dir=model_dir,
)
'''
auto_loader = AutoLoader(
    "seq2seq",
    model_name="gpm-13b",
    model_dir=model_dir,
)
'''
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()

trainer.pre_train(model)

### lambada
data_prefix = '/home/ldwang/Downloads/pile/lambada_text_document'
data_impl = 'mmap'
splits_string = '10000,0,0'
train_valid_test_num_samples = [5153, 0, 0]
seq_length = 1024
seq_length = 2048
seed = 2023
skip_warmup = True

### wikitext gpm_13b
data_prefix = '/home/ldwang/Downloads/pile/wikitext_text_document'
data_impl = 'mmap'
splits_string = '10000,0,0'
train_valid_test_num_samples = [2891, 0, 0]
seq_length = 2048
seed = 2023
skip_warmup = True

### wikitext gpt2_base
data_prefix = '/home/ldwang/Downloads/pile/wikitext_text_document'
data_impl = 'mmap'
splits_string = '10000,0,0'
train_valid_test_num_samples = [2891, 0, 0]
seq_length = 1024
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

sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=trainer.world_size,
    rank=trainer.rank,
    shuffle=False)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=env_args.batch_size,
    sampler=sampler,
    collate_fn=collate_fn,
    num_workers=4,
    prefetch_factor=4,
    pin_memory=True,
    drop_last=False,
    shuffle=False)

model.eval()

ppls = 0
accs = 0
lens = 0
losses = 0
for iteration, batch in enumerate(train_dataloader):
    device = next(model.parameters()).device
    data = {
        x: batch[x].to(torch.device(device))
        for x in batch if x not in ['uid', 'meta', 'mode']}
    print(f"rank={trainer.rank}, iteration={iteration}")

    with torch.no_grad():
        step_output = trainer.forward_step(data, model, mems=None)
        trg_len = data['labels'].shape[-1]
        trg_len_minus_one = trg_len - 1

        labels = data['labels']
        logits = step_output['logits']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        predicted = torch.argmax(shift_logits, -1)
        acc = (predicted==shift_labels).float().sum()/trg_len_minus_one

        lm_loss = step_output['loss']
        reduced_loss = lm_loss.detach().clone().view(1)
        ppl = math.exp(reduced_loss)

        losses += reduced_loss.item() * trg_len_minus_one
        ppls += ppl * trg_len_minus_one
        accs += acc * trg_len_minus_one
        lens += trg_len_minus_one

avg_ppl = ppls / lens
print(f"rank={trainer.rank}, ppls={ppls}, lens={lens}, avg={avg_ppl}")
avg_acc = accs / lens
print(f"rank={trainer.rank}, accs={accs}, lens={lens}, avg={avg_acc}")
avg_loss = losses / lens
print(f"rank={trainer.rank}, losses={losses}, lens={lens}, avg={avg_loss}")
