# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch 
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSeq2Seq
from flagai.data.tokenizer import Tokenizer
from flagai.data.dataset import Seq2SeqDataset
from flagai.test_utils import Seq2SeqCollateArguments
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, CH_TASKS
from flagai.data.dataset import ConstructSeq2seqStrategy
from flagai.metrics import accuracy_metric, exact_match_score,bleu_metric, rouge_metric

# Compared with original seq2seq, seq2seq dataset is used
# task_name :['cmrc',xxxx]
task_name = "cnn_dm"

cl_args = Seq2SeqCollateArguments()

print("downloading...")

if task_name in CH_TASKS:
    model_name = 'GLM-large-ch'
else:
    model_name = 'GLM-large-en'

tokenizer = Tokenizer.from_pretrained(model_name)

train_dataset = Seq2SeqDataset(task_name=task_name,
                               data_dir='../../datasets/',
                               dataset_type='test',
                               tokenizer=tokenizer)
valid_dataset = Seq2SeqDataset(task_name=task_name,
                               data_dir='../../datasets/',
                               dataset_type='test',
                               tokenizer=tokenizer)
collate_fn = ConstructSeq2seqStrategy(cl_args,
                                      tokenizer,
                                      task_name=task_name)

model = GLMForSeq2Seq.from_pretrain(model_name=model_name)
model.load_state_dict(torch.load("/home/yanzhaodong/anhforth/FlagAI/examples/glm_seq2seq/checkpoints/140000/pytorch_model.bin")["module"])
trainer = Trainer(env_type='deepspeed',
                  epochs=10000000,
                  batch_size=16,
                  gradient_accumulation_steps=5,
                  checkpoint_activations=True,
                  eval_interval=False,
                  log_interval=100,
                  fp16=True,
                  save_interval=10000,
                  experiment_name='glm_large',
                  load_dir=None,
                  num_nodes=1,
                  num_gpus=2,
                  tokenizer=tokenizer,
                  hostfile='./hostfile',
                  deepspeed_config='./deepspeed.json',
                  lr=1e-4,
                  training_script=__file__)
# optimizer = Adam(param_groups,
#                              lr=1e-3,
#                              weight_decay=0,
#                              betas=(0.9, 0.999),
#                              eps=1e-8)
trainer.train(model,
              collate_fn=collate_fn,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              metric_methods=[])
