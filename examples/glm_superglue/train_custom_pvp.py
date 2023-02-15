# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import sys 
sys.path.append("/home/yanzhaodong/anhforth/test/FlagAI")
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSequenceClassification
from flagai.model.glm_model import GLMForSingleTokenCloze
from flagai.data.tokenizer import Tokenizer

from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, MULTI_TOKEN_TASKS, CH_TASKS
from flagai.data.dataset import ConstructSuperglueStrategy
from flagai.data.dataset.superglue.pvp import PVP
from flagai.data.dataset.data_utils import build_input_from_ids, build_sample, InputExample
from flagai.data.dataset.data_utils import build_decoder_input, build_decoder_sample, num_special_tokens_to_add
from typing import Tuple, List, Union, Dict
import string

class RtePVP(PVP):
    VERBALIZER = {"not_entailment": [" No"], "entailment": [" Yes"]}

    @staticmethod
    def available_patterns():
        return [0, 1, 2, 3, 4]

    @property
    def spell_length(self):
        return self.num_prompt_tokens + self.prefix_prompt

    def get_parts(self, example: InputExample):
        # switch text_a and text_b to get the correct order
        text_a = example.text_a
        text_b = example.text_b.rstrip(string.punctuation)
        if self.pattern_id == 0:
            parts_a, parts_b = [None, '"',
                                self.shortenable(text_b), '" ?'], [
                                    None, [self.mask], ',', None, ' "',
                                    self.shortenable(text_a), '"'
                                ]
        elif self.pattern_id == 1:
            parts_a, parts_b = [None, self.shortenable(text_b), '?'], [
                None, [self.mask], ',', None,
                self.shortenable(" " + text_a)
            ]
        elif self.pattern_id == 2:
            parts_a, parts_b = [
                None,
                self.shortenable(text_a), None, ' question:',
                self.shortenable(" " + text_b), ' True or False?', None,
                ' answer:', [self.mask]
            ], []
        else:
            raise NotImplementedError(self.pattern_id)
        parts_a, parts_b = self.replace_prompt_tokens(parts_a, parts_b)
        return parts_a, parts_b

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 4:
            return [' true'] if label == 'entailment' else [' false']
        return RtePVP.VERBALIZER[label]


# task_name options: ['boolq', 'cb', 'copa', 'multirc', 'rte', 'wic', 'wsc', 'afqmc', 'tnews']
task_name = "rte"

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
cl_args.cloze_eval = True
cl_args.multi_token = task_name in MULTI_TOKEN_TASKS

cl_args.continuous_prompt = True
cl_args.prefix_prompt = 2
cl_args.num_prompt_tokens = 5

if task_name in CH_TASKS:
    model_name = 'GLM-large-ch'
    add_block_symbols=True,
else:
    model_name = 'GLM-large-en'
tokenizer = Tokenizer.from_pretrained(model_name)

# model = GLMForSequenceClassification.from_pretrain(model_name=model_name, spell_length=2,
#                                                     class_num=3, tune_prefix_layers=1)

model = GLMForSingleTokenCloze.from_pretrain(download_path="./checkpoints",
                                             model_name=model_name, spell_length=2,
                                            class_num=3, tune_prefix_layers=1)
train_dataset = SuperGlueDataset(task_name=task_name,
                                    data_dir='./datasets/',
                                    dataset_type='train',
                                    tokenizer=tokenizer)

collate_fn = ConstructSuperglueStrategy(cl_args,
                                        tokenizer,
                                        task_name=task_name,
                                        PET=True,
                                        custom_pvp=RtePVP)

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

