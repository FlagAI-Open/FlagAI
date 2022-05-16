from flagai.model.gpt2_model import GPT2Model
from flagai.data.tokenizer.bert.bert_tokenizer import BertTokenizer
import torch
import unittest
import os

class GPT2TestCase(unittest.TestCase):
    def setUp(self) -> None:

        self.model = GPT2Model.init_from_json("./checkpoints/gpt2_base_chinese/config.json")
        self.tokenizer = BertTokenizer("./checkpoints/gpt2_base_chinese/vocab.txt")

        print("loading model successfully!")

    def test_model_predict(self):
        input_ids = self.tokenizer.encode_plus("今天吃饭吃了肯德基")["input_ids"]

        input_ids = torch.LongTensor([input_ids])

        output = self.model(input_ids=input_ids)
        print(output)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(GPT2TestCase('test_model_predict'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())