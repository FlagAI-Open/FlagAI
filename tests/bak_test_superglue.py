# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSingleTokenCloze, GLMForMultiTokenCloze, GLMForSequenceClassification
from flagai.data.tokenizer import GLMLargeEnWordPieceTokenizer, GLMLargeChTokenizer, GLM10bENBPETokenizer
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, MULTI_TOKEN_TASKS, CH_TASKS
import unittest
from flagai.data.dataset import ConstructSuperglueStrategy
from flagai.data.tokenizer import BertWordPieceTokenizer

class TrainerTestCase(unittest.TestCase):

    def test_init_trainer_pytorch(self):
        from flagai.auto_model.auto_loader import AutoLoader

        auto_loader = AutoLoader(
            task_name="title-generation",
            model_name="BERT-base-en"
        )
        model = auto_loader.get_model()
        tokenizer = auto_loader.get_tokenizer()

        from flagai.model.predictor.predictor import Predictor
        predictor = Predictor(model, tokenizer)
        test_data = [
            "Four minutes after the red card, Emerson Royal nodded a corner into the path of the unmarked Kane at the far post, who nudged the ball in for his 12th goal in 17 North London derby appearances. Arteta's misery was compounded two minutes after half-time when Kane held the ball up in front of goal and teed up Son to smash a shot beyond a crowd of defenders to make it 3-0.The goal moved the South Korea talisman a goal behind Premier League top scorer Mohamed Salah on 21 for the season, and he looked perturbed when he was hauled off with 18 minutes remaining, receiving words of consolation from Pierre-Emile Hojbjerg.Once his frustrations have eased, Son and Spurs will look ahead to two final games in which they only need a point more than Arsenal to finish fourth.",
        ]

        for text in test_data:
            print(
                predictor.predict_generate_beamsearch(text,
                                                      out_max_length=50,
                                                      beam_size=3))
        # # for task_name in [
        # #         'boolq', 'cb', 'copa', 'multirc', 'rte', 'wic', 'wsc', 'afqmc',
        # #         'tnews', 'qqp', 'cola', 'mnli', 'qnli'
        # # ]:
        # for task_name in [
        #     'qqp'
        # ]:
        #     trainer = Trainer(env_type='pytorch',
        #                       epochs=1,
        #                       batch_size=4,
        #                       eval_interval=100,
        #                       log_interval=50,
        #                       experiment_name='glm_large',
        #                       pytorch_device='cuda',
        #                       load_dir=None,
        #                       fp16=True,
        #                       lr=1e-5)
        #     print("downloading...")
        #
        #     cl_args = CollateArguments()
        #     cl_args.cloze_eval = False
        #     cl_args.multi_token = task_name in MULTI_TOKEN_TASKS
        #     if task_name in CH_TASKS:
        #         model_name = 'GLM-large-ch'
        #         tokenizer = GLMLargeChTokenizer()
        #     else:
        #         model_name = 'GLM-large-en'
        #         tokenizer = GLMLargeEnWordPieceTokenizer()
        #         tokenizer = GLM10bENBPETokenizer(add_block_symbols=True)
        #
        #     if cl_args.cloze_eval:
        #         if cl_args.multi_token:
        #             model = GLMForMultiTokenCloze.from_pretrain(
        #                 model_name=model_name, only_download_config=True)
        #         else:
        #             model = GLMForSingleTokenCloze.from_pretrain(
        #                 model_name=model_name, only_download_config=True)
        #     else:
        #         model = GLMForSequenceClassification.from_pretrain(
        #             model_name=model_name, only_download_config=True, class_num=2)
        #
        #     train_dataset = SuperGlueDataset(task_name=task_name,
        #                                      data_dir='./datasets/',
        #                                      dataset_type='train',
        #                                      tokenizer=tokenizer)
        #     train_dataset.example_list = train_dataset.example_list[:1]
        #     collate_fn = ConstructSuperglueStrategy(cl_args,
        #                                             tokenizer,
        #                                             task_name=task_name)
        #
        #     valid_dataset = SuperGlueDataset(task_name=task_name,
        #                                      data_dir='./datasets/',
        #                                      dataset_type='dev',
        #                                      tokenizer=tokenizer)
        #     valid_dataset.example_list = valid_dataset.example_list[:1]
        #
        #     metric_methods = DEFAULT_METRICS[task_name]
        #     trainer.train(model,
        #                   collate_fn=collate_fn,
        #                   train_dataset=train_dataset,
        #                   valid_dataset=valid_dataset,
        #                   metric_methods=metric_methods)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TrainerTestCase('test_init_trainer_pytorch'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
