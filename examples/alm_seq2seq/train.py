# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.trainer import Trainer
from flagai.model.alm_model import ALMForSeq2Seq
from flagai.data.tokenizer import Tokenizer
from flagai.data.dataset import Seq2SeqDataset
from flagai.test_utils import Seq2SeqCollateArguments
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, CH_TASKS
from flagai.data.dataset import ConstructSeq2seqStrategy


# Compared with original seq2seq, seq2seq dataset is used
# task_name :['cmrc',xxxx]
task_name = "cmrc"

cl_args = Seq2SeqCollateArguments()
trainer = Trainer(env_type='pytorch',
                  epochs=1,
                  batch_size=4,
                  eval_interval=5,
                  log_interval=50,
                  experiment_name='glm_large',
                  pytorch_device='cuda',
                  load_dir=None,
                  lr=1e-4)
print("downloading...")

if task_name in CH_TASKS:
    model_name = 'ALM-1.0'
else:
    model_name = 'GLM-large-en'

print('model_name', model_name)
tokenizer = Tokenizer.from_pretrained(model_name)
model = ALMForSeq2Seq.from_pretrain(model_name=model_name)


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
train_dataset.example_list = train_dataset.example_list[:20]
valid_dataset.example_list = valid_dataset.example_list[:20]



trainer.train(model,
              collate_fn=collate_fn,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              metric_methods=[])
