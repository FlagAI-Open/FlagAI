# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSingleTokenCloze, GLMForMultiTokenCloze, GLMForSequenceClassification
from flagai.data.tokenizer import Tokenizer
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, MULTI_TOKEN_TASKS, CH_TASKS
import unittest
from flagai.data.dataset import ConstructSuperglueStrategy


class TrainerTestCase(unittest.TestCase):

    def test_init_trainer_pytorch(self):
        for task_name in [
                'boolq', 'cb', 'copa', 'multirc', 'rte', 'wic', 'wsc', 'afqmc',
                'tnews', 'qqp', 'cola', 'mnli', 'qnli'
        ]:
            trainer = Trainer(env_type='pytorch',
                              epochs=1,
                              batch_size=4,
                              eval_interval=100,
                              log_interval=50,
                              experiment_name='glm_large',
                              pytorch_device='cuda',
                              load_dir=None,
                              fp16=True,
                              lr=1e-5)
            print("downloading...")

            cl_args = CollateArguments()
            cl_args.cloze_eval = True
            cl_args.multi_token = task_name in MULTI_TOKEN_TASKS
            if task_name in CH_TASKS:
                model_name = 'GLM-large-ch'
            else:
                model_name = 'GLM-large-en'

            tokenizer = Tokenizer.from_pretrained(model_name)
            if cl_args.cloze_eval:
                if cl_args.multi_token:
                    model = GLMForMultiTokenCloze.from_pretrain(
                        model_name=model_name, only_download_config=True)
                else:
                    model = GLMForSingleTokenCloze.from_pretrain(
                        model_name=model_name, only_download_config=True)
            else:
                model = GLMForSequenceClassification.from_pretrain(
                    model_name=model_name, only_download_config=True, class_num=2)

            train_dataset = SuperGlueDataset(task_name=task_name,
                                             data_dir='./datasets/',
                                             dataset_type='train',
                                             tokenizer=tokenizer)
            train_dataset.example_list = train_dataset.example_list[:1]
            collate_fn = ConstructSuperglueStrategy(cl_args,
                                                    tokenizer,
                                                    task_name=task_name)

            valid_dataset = SuperGlueDataset(task_name=task_name,
                                             data_dir='./datasets/',
                                             dataset_type='dev',
                                             tokenizer=tokenizer)
            valid_dataset.example_list = valid_dataset.example_list[:1]

            metric_methods = DEFAULT_METRICS[task_name]
            trainer.train(model,
                          collate_fn=collate_fn,
                          train_dataset=train_dataset,
                          valid_dataset=valid_dataset,
                          metric_methods=metric_methods)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TrainerTestCase('test_init_trainer_pytorch'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())