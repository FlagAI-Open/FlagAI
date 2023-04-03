# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSingleTokenCloze
from flagai.data.tokenizer import Tokenizer
from flagai.metrics import accuracy_metric
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments

task_name = 'qqp'
trainer = Trainer(env_type='pytorchDDP',
                  epochs=2,
                  batch_size=16,
                  eval_interval=20,
                  log_interval=2,
                  save_interval = 1e5,
                  gradient_accumulation_steps=2,
                  checkpoint_activations=True,
                  lr=2e-4,
                  fp16=True,
                  warm_up=1.0,
                  weight_decay=0.1,
                  save_dir="./qqp",
                  master_ip='127.0.0.1',
                  master_port=16665,
                  num_nodes=1,
                  num_gpus=2,
                  hostfile='./hostfile',
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
cl_args.cloze_eval = True


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
