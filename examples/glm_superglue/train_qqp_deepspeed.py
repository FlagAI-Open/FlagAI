# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import sys
sys.path.append('/mnt/liuguang/FlagAI')
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSingleTokenCloze
from flagai.data.tokenizer import GLM10bENBPETokenizer, GLMLargeEnWordPieceTokenizer
from flagai.metrics import accuracy_metric
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments

task_name = 'qqp'
trainer = Trainer(env_type='deepspeed',
                  epochs=2,
                  batch_size=512,
                  eval_interval=100,
                  log_interval=10,
                  save_interval = 1e5,
                  gradient_accumulation_steps=5,
                  checkpoint_activations=True,
                  fp16=True,
                  warm_up=0.1,
                  weight_decay=0.1,
                  save_dir="./qqp",
                  master_ip='127.0.0.1',
                  master_port=17887,
                  num_nodes=1,
                  num_gpus=2,
                  hostfile='./hostfile',
                  deepspeed_config='./deepspeed.json',
                  training_script=__file__)

model = GLMForSingleTokenCloze.from_pretrain(download_path="/mnt/test_10b_models",
                                             model_name="GLM-large-en")

#tokenizer = GLM10bENBPETokenizer()
tokenizer = GLMLargeEnWordPieceTokenizer()
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
# train_dataset.example_list = train_dataset.example_list[:160]
# valid_dataset.example_list = valid_dataset.example_list[:160]


trainer.train(model,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              collate_fn=collate_fn,
              metric_methods=[["acc", accuracy_metric]])
