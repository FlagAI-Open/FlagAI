# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.auto_model.auto_loader import AutoLoader
import unittest

class AutoLoaderTestCase(unittest.TestCase):

    def setUp(self) -> None:

        self.task_name = [
            "seq2seq", "ner", "classification","poetry", "lm",
            "title-generation", "semantic-matching", "embedding"
        ]
        self.model_name = [
            "GPT2-base-ch", "T5-base-ch", "T5-base-en","RoBERTa-base-ch",
            "GLM-large-ch","GLM-large-cn", "BERT-base-en", "opt-350m-en",
            "opt-125m-en", "opt-1.3b-en", "opt-2.7b-en",
        ]


    def test_GLM_large_en(self):
        for t_name  in self.task_name:
            m_name = 'GLM-large-en'
            
            loader = AutoLoader(task_name=t_name,
                                model_name=m_name,
                                class_num=3,
                                inner_dim=32,
                                only_download_config=True)
            print(
                f"task_name is {t_name}, model_name is {m_name}"
            )
    def test_GLM_large_ch(self):
        for t_name  in self.task_name:
            m_name = 'GLM-large-ch'
            
            loader = AutoLoader(task_name=t_name,
                                model_name=m_name,
                                class_num=3,
                                inner_dim=32,
                                only_download_config=True)
            print(
                f"task_name is {t_name}, model_name is {m_name}"
            )
    def test_BERT_base_en(self):
        for t_name  in self.task_name:
            m_name = 'BERT-base-en'
            
            loader = AutoLoader(task_name=t_name,
                                model_name=m_name,
                                class_num=3,
                                inner_dim=32,
                                only_download_config=True)
            print(
                f"task_name is {t_name}, model_name is {m_name}"
            )
    def test_RoBERTa_base_ch(self):
        for t_name  in self.task_name:
            m_name = 'RoBERTa-base-ch'
            
            loader = AutoLoader(task_name=t_name,
                                model_name=m_name,
                                class_num=3,
                                inner_dim=32,
                                only_download_config=True)
            print(
                f"task_name is {t_name}, model_name is {m_name}"
            )

    def test_GPT2_base_ch(self):
        for t_name  in self.task_name:
            m_name = 'GPT2-base-ch'
            
            loader = AutoLoader(task_name=t_name,
                                model_name=m_name,
                                class_num=3,
                                inner_dim=32,
                                only_download_config=True)
            print(
                f"task_name is {t_name}, model_name is {m_name}"
            )
    def test_T5_base_ch(self):
        for t_name in self.task_name:
            m_name = 'T5-base-ch'
            loader = AutoLoader(task_name=t_name,
                                model_name=m_name,
                                class_num=3,
                                inner_dim=32,
                                only_download_config=True)
            print(
                f"task_name is {t_name}, model_name is {m_name}"
            )

    def test_CPM_large_ch(self):
        for t_name in self.task_name:
            m_name = 'CPM-large-ch-generation'
            loader = AutoLoader(task_name=t_name,
                                model_name=m_name,
                                class_num=3,
                                inner_dim=32,
                                only_download_config=True)
            print(
                f"task_name is {t_name}, model_name is {m_name}"
            )

    def test_OPT_model(self):

        for m_name in ["opt-350m-en", "opt-125m-en", "opt-1.3b-en", "opt-2.7b-en"]:
            loader = AutoLoader(task_name="lm",
                                model_name=m_name,
                                only_download_config=True)
            print(
                f"task_name is lm, model_name is {m_name}"
            )

    def test_EVA_CLIP_model(self):
        loader = AutoLoader(task_name="txt_img_matching", 
                            model_name="eva-clip")
        print(
                "task_name is txt_img_matching, model_name is eva-clip"
            )
   
def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoLoaderTestCase('test_GLM_large_ch'))
    suite.addTest(AutoLoaderTestCase('test_GLM_large_en'))
    suite.addTest(AutoLoaderTestCase('test_BERT_base_en'))
    suite.addTest(AutoLoaderTestCase('test_RoBERTa_base_ch'))
    suite.addTest(AutoLoaderTestCase('test_T5_base_ch'))
    suite.addTest(AutoLoaderTestCase('test_GPT2_base_ch'))
    suite.addTest(AutoLoaderTestCase('test_CPM_large_ch'))
    suite.addTest(AutoLoaderTestCase('test_OPT_model'))
    suite.addTest(AutoLoaderTestCase('test_EVA_CLIP_model'))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
