# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSingleTokenCloze
from flagai.data.tokenizer import Tokenizer
from flagai.metrics import accuracy_metric
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments


task_name = 'cb'
trainer = Trainer(env_type='pytorch',
                 pytorch_device="cuda",
                  epochs=1000000000,
                  batch_size=32,
                  eval_interval=100000000,
                  checkpoint_activations=False,
                  fp16=True,
                  save_interval=10000,
                  log_interval=50,
                  save_dir="./glm_superglue_en",
                  master_ip='127.0.0.1',
                  master_port=17755,
                  num_nodes=1,
                  num_gpus=2,
                  hostfile='./hostfile',
                  model_parallel_size=2,
                  deepspeed_config='./deepspeed.json',
                  lr=1e-4,
                  training_script=__file__)

model_name = "GLM-large-en"
model = GLMForSingleTokenCloze.from_pretrain(download_path="./checkpoints",
                                             model_name=model_name)
                 
tokenizer = Tokenizer.from_pretrained(model_name)
train_dataset = SuperGlueDataset(task_name=task_name,
                                 data_dir='./datasets/',
                                 dataset_type='train',
                                 tokenizer=tokenizer,
                                 cloze_eval=True)
valid_dataset = SuperGlueDataset(task_name=task_name,
                                 data_dir='./datasets/',
                                 dataset_type='dev',
                                 tokenizer=tokenizer,
                                 cloze_eval=True)

cl_args = CollateArguments()

if task_name in ['copa', 'wsc', 'record']:
    cl_args.multi_token = True

from flagai.data.dataset import ConstructSuperglueStrategy

collate_fn = ConstructSuperglueStrategy(cl_args,
                                        tokenizer,
                                        task_name=task_name)
trainer.train(model,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              collate_fn=collate_fn,
              metric_methods=[["acc", accuracy_metric]])
