# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSeq2Seq
from flagai.data.tokenizer import Tokenizer
from flagai.data.dataset import Seq2SeqDataset
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, CH_TASKS
from flagai.data.dataset import ConstructSeq2seqStrategy
from flagai.test_utils import Seq2SeqCollateArguments
import unittest


class TrainerTestCase(unittest.TestCase):

    def test_init_trainer_pytorch(self):
        # Compared with original seq2seq, seq2seq dataset is used
        # task_name :['cmrc',xxxx]
        task_name = "cmrc"
        cl_args = Seq2SeqCollateArguments()
        trainer = Trainer(env_type='pytorch',
                          epochs=1,
                          batch_size=1,
                          eval_interval=100,
                          log_interval=50,
                          experiment_name='glm_large',
                          fp16=True,
                          pytorch_device='cuda',
                          lr=1e-4)
        print("downloading...")

        if task_name in CH_TASKS:
            model_name = 'GLM-large-ch'
        else:
            model_name = 'GLM-large-en'
        tokenizer = Tokenizer.from_pretrained(model_name)

        train_dataset = Seq2SeqDataset(task_name=task_name,
                                       data_dir='./data/',
                                       dataset_type='train',
                                       tokenizer=tokenizer)
        valid_dataset = Seq2SeqDataset(task_name=task_name,
                                       data_dir='./data/',
                                       dataset_type='dev',
                                       tokenizer=tokenizer)

        collate_fn = ConstructSeq2seqStrategy(cl_args,
                                              tokenizer,
                                              task_name=task_name)

        train_dataset.example_list = train_dataset.example_list[:10]
        valid_dataset.example_list = valid_dataset.example_list[:10]

        model = GLMForSeq2Seq.from_pretrain(model_name=model_name,
                                            only_download_config=True)

        trainer.train(model,
                      collate_fn=collate_fn,
                      train_dataset=train_dataset,
                      valid_dataset=valid_dataset,
                      metric_methods=DEFAULT_METRICS[task_name])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TrainerTestCase('test_init_trainer_pytorch'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
