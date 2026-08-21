# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
import torch
from flagai.data.tokenizer import Tokenizer
from flagai.model.bert_model import BertModel, BertForSeq2seq, \
                                    BertForSequenceLabeling, \
                                    BertForSequenceLabelingGP, \
                                    BertForClsClassifier, \
                                    BertForSequenceLabelingCRF
from flagai.data.tokenizer.bert.bert_tokenizer import BertTokenizer
import unittest

class BertTestCase(unittest.TestCase):

    def setUp(self) -> None:

        self.models = [BertForClsClassifier,
                       BertForSeq2seq,
                       BertForSequenceLabeling,
                       BertForSequenceLabelingGP,
                       BertForSequenceLabelingCRF]
        self.model_name = "RoBERTa-base-ch"
        self.bert_path = "./checkpoints/RoBERTa-base-ch/config.json"
        # self.tokenizer = BertTokenizer("./checkpoints/RoBERTa-base-ch/vocab.txt")
        self.tokenizer = Tokenizer.from_pretrained(self.model_name)
        print("loading bert model successfully!")

    def test_model_predict(self):

        for model in self.models:
            model = model.init_from_json(self.bert_path, class_num=3, inner_dim=64)
            class_name = type(model).__name__.lower()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            predictor = Predictor(model, self.tokenizer)
            text = "今天吃饭吃了肯德基"
            if "seq2seq" in class_name:
                output = predictor.predict_generate_beamsearch(text, out_max_length=20)
                output = predictor.predict_generate_randomsample(text, out_max_length=20)
            elif "cls" in class_name:
                output = predictor.predict_cls_classifier(text)
            elif "sequencelabeling" in class_name:
                output = predictor.predict_ner(text, target=["0", "1", "2"])
            else :
                output = None
            print(f"model_name is {class_name}, output is {output}")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(BertTestCase('test_model_predict'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
