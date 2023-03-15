# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSequenceClassification
from flagai.model.glm_model import GLMForSingleTokenCloze
from flagai.data.tokenizer import Tokenizer
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, MULTI_TOKEN_TASKS, CH_TASKS
from flagai.data.dataset import ConstructSuperglueStrategy


# task_name options: ['boolq', 'cb', 'copa', 'multirc', 'rte', 'wic', 'wsc', 'afqmc', 'tnews']
task_name = "qqp"

cl_args = CollateArguments()
cl_args.multi_token = task_name in MULTI_TOKEN_TASKS
if task_name in CH_TASKS:
    model_name = 'GLM-large-ch'
    add_block_symbols=True
else:
    model_name = 'GLM-large-en'
tokenizer = Tokenizer.from_pretrained(model_name)
model = GLMForSingleTokenCloze.from_pretrain(download_path="./checkpoints",
                                             model_name=model_name)


# Continue training from saved checkpoints
# model_save_path = "./checkpoints/20000/pytorch_model.bin"
# model.load_state_dict(torch.load(model_save_path, map_location="cuda")["module"])  

train_dataset = SuperGlueDataset(task_name=task_name,
                                    data_dir='./datasets/',
                                    dataset_type='train',
                                    tokenizer=tokenizer)

collate_fn = ConstructSuperglueStrategy(cl_args,
                                        tokenizer,
                                        task_name=task_name)

valid_dataset = SuperGlueDataset(task_name=task_name,
                                    data_dir='./datasets/',
                                    dataset_type='dev',
                                    tokenizer=tokenizer)

metric_methods = DEFAULT_METRICS[task_name]

trainer = Trainer(env_type='deepspeed',
                  epochs=1000,
                  batch_size=64,
                  gradient_accumulation_steps=5,
                  checkpoint_activations=True,
                  eval_interval=1000,
                  log_interval=100,
                  fp16=True,
                  save_interval=10000,
                  experiment_name='glm_large',
                  load_dir=None,
                  num_nodes=1,
                  num_gpus=2,
                  hostfile='./hostfile',
                  deepspeed_config='./deepspeed.json',
                  lr=1e-4,
                  training_script=__file__)

trainer.train(model,
                collate_fn=collate_fn,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                metric_methods=metric_methods)

