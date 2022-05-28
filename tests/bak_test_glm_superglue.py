# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMModel, GLMForSequenceClassification, GLMForSingleTokenCloze, GLMForMultiTokenCloze, GLMForMultiTokenClozeFast
from flagai.data.tokenizer import GLMLargeEnWordPieceTokenizer, GLMLargeChTokenizer
from flagai.metrics import *

from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, MULTI_TOKEN_TASKS, CH_TASKS
import unittest
from flagai.data.dataset import ConstructSuperglueStrategy


class TrainerTestCase(unittest.TestCase):

    def test_init_trainer_pytorch(self):
        for task_name in [
                'boolq', 'cb', 'copa', 'multirc', 'rte', 'wic', 'wsc', 'afqmc',
                'tnews'
        ]:
            trainer = Trainer(env_type='pytorch',
                              epochs=1,
                              batch_size=4,
                              eval_interval=100,
                              log_interval=50,
                              experiment_name='glm_large',
                              pytorch_device='cpu',
                              load_dir=None,
                              lr=1e-4,
                              save_epoch=10)
            print("downloading...")

            cl_args = CollateArguments()
            cl_args.multi_token = task_name in MULTI_TOKEN_TASKS
            model_name = 'GLM-large-ch'
            if task_name in CH_TASKS:
                model_name = 'GLM-large-ch'
                #lm_model = GLMModel.from_pretrain(model_name='GLM-large-ch')
                tokenizer = GLMLargeChTokenizer(add_block_symbols=True,
                                                add_task_mask=False,
                                                add_decoder_mask=False,
                                                fix_command_token=True)
                # tokenizer = GLMBertWordPieceTokenizer(tokenizer_model_type='BERT-base-ch')
            else:
                model_name = 'GLM-large-en'
                #lm_model = GLMModel.from_pretrain(model_name='GLM-large-en')
                tokenizer = GLMLargeEnWordPieceTokenizer()

            if cl_args.multi_token:
                model = GLMForMultiTokenCloze.from_pretrain(
                    model_name=model_name, only_download_config=True)
            else:
                model = GLMForSingleTokenCloze.from_pretrain(
                    model_name=model_name, only_download_config=True)

            train_dataset = SuperGlueDataset(task_name=task_name,
                                             data_dir='./datasets/',
                                             dataset_type='train',
                                             tokenizer=tokenizer)
            train_dataset.example_list = train_dataset.example_list[:2]

            collate_fn = ConstructSuperglueStrategy(cl_args,
                                                    tokenizer,
                                                    task_name=task_name)

            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=16,
                                                       shuffle=False,
                                                       num_workers=1,
                                                       drop_last=False,
                                                       pin_memory=False,
                                                       collate_fn=collate_fn)
            for data_iterator in train_loader:
                dct = data_iterator
                '''
                input_ids torch.Size([16, 2, 256])
                labels torch.Size([16])
                position_ids torch.Size([16, 2, 2, 256])
                attention_mask torch.Size([16, 2])
                target_ids torch.Size([16, 2, 256])
                logit_mask torch.Size([16, 2, 256])
                loss_mask torch.Size([16, 2])
  
                '''
                for key, value in dct.items():
                    try:
                        print(key, value.size())
                    except:
                        print(key, len(value))
                print(dct['labels'])
                break
            valid_dataset = SuperGlueDataset(task_name=task_name,
                                             data_dir='./datasets/',
                                             dataset_type='dev',
                                             tokenizer=tokenizer)
            valid_dataset.example_list = valid_dataset.example_list[:2]

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
