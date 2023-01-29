# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import sys 
sys.path.append("/home/yanzhaodong/anhforth/FlagAI")
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
task_name = "cmrc"

cl_args = Seq2SeqCollateArguments()

print("downloading...")

if task_name in CH_TASKS:
    model_name = 'GLM-large-ch'
else:
    model_name = 'GLM-large-en'

tokenizer = Tokenizer.from_pretrained(model_name)

train_dataset = Seq2SeqDataset(task_name=task_name,
                               data_dir='./datasets/',
                               dataset_type='train',
                               tokenizer=tokenizer)
valid_dataset = Seq2SeqDataset(task_name=task_name,
                               data_dir='./datasets/',
                               dataset_type='dev',
                               tokenizer=tokenizer)                            
collate_fn = ConstructSeq2seqStrategy(cl_args,
                                      tokenizer,
                                      task_name=task_name)

model = GLMForSeq2Seq.from_pretrain(model_name=model_name)
# model.load_state_dict(torch.load("./checkpoints/2000/pytorch_model.bin")["module"])


trainer = Trainer(env_type='pytorch',
                  epochs=0,
                  batch_size=1,
                  eval_interval=10,
                  log_interval=50,
                  experiment_name='glm_large',
                  pytorch_device='cuda',
                  load_dir=None,
                  tokenizer=tokenizer,
                  lr=1e-4)

trainer.train(model,
              collate_fn=collate_fn,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset[:100],
              metric_methods=[["blue_metric", bleu_metric],["rouge_metric", rouge_metric]])
