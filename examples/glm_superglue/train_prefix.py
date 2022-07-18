# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSequenceClassification
from flagai.data.tokenizer import Tokenizer

from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, MULTI_TOKEN_TASKS, CH_TASKS
from flagai.data.dataset import ConstructSuperglueStrategy


# task_name options: ['boolq', 'cb', 'copa', 'multirc', 'rte', 'wic', 'wsc', 'afqmc', 'tnews']
task_name = "cb"

trainer = Trainer(env_type='pytorch',
                    epochs=10,
                    batch_size=4,
                    eval_interval=100,
                    log_interval=50,
                    experiment_name='glm_large',
                    pytorch_device='cuda',
                    load_dir=None,
                    lr=1e-4)
print("downloading...")

cl_args = CollateArguments()
cl_args.cloze_eval = False
cl_args.multi_token = task_name in MULTI_TOKEN_TASKS

if task_name in CH_TASKS:
    model_name = 'GLM-large-ch'
    add_block_symbols=True,
else:
    model_name = 'GLM-large-en'
tokenizer = Tokenizer.from_pretrained(model_name)

model = GLMForSequenceClassification.from_pretrain(model_name=model_name, spell_length=2,
                                                    class_num=3, tune_prefix_layers=1)

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
trainer.train(model,
                collate_fn=collate_fn,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                metric_methods=metric_methods)

