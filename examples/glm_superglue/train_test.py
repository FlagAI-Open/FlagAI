# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSingleTokenCloze
from flagai.data.tokenizer import GLM10bENBPETokenizer, GLMLargeEnWordPieceTokenizer
from flagai.metrics import accuracy_metric
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments



task_name = 'qqp'
trainer = Trainer(env_type='deepspeed',
                  pytorch_device='cuda:1',
                  epochs=10,
                  batch_size=36,
                  eval_interval=80,
                  log_interval=4,
                  checkpoint_activations=False,
                  fp16=True,
                  warm_up=0.1,
                  save_dir="./glm_superglue_en",
                  master_ip='127.0.0.1',
                  master_port=17755,
                  num_nodes=1,
                  num_gpus=4,
                  hostfile='./hostfile',
                  model_parallel_size=1,
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


trainer.train(model,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              collate_fn=collate_fn,
              metric_methods=[["acc", accuracy_metric]])
