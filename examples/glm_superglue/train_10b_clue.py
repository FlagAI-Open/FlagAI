# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSingleTokenCloze
from flagai.data.tokenizer import Tokenizer
from flagai.metrics import accuracy_metric
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments
from flagai.data.dataset import ConstructSuperglueStrategy

task_name = 'afqmc'
trainer = Trainer(env_type="pytorch",
                  batch_size=16,
                  epochs=10,
                  log_interval=100,
                  eval_interval=500,
                  load_dir=None,
                  pytorch_device="cuda",
                  save_dir="./glm_superglue_en",
                  save_interval=1)

model_name = "GLM-large-ch"
model = GLMForSingleTokenCloze.from_pretrain(download_path="./checkpoints",
                                             model_name="GLM-large-ch")


tokenizer = Tokenizer.from_pretrained("GLM-large-ch")
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
cl_args.multi_token = False

collate_fn = ConstructSuperglueStrategy(cl_args,
                                        tokenizer,
                                        task_name=task_name)
trainer.train(model,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              collate_fn=collate_fn,
              metric_methods=[["acc", accuracy_metric]])
