# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.model.t5_model import T5Model
from flagai.data.tokenizer.t5.t5_pegasus_tokenizer import T5PegasusTokenizer
from flagai.model.predictor.predictor import Predictor
import unittest

class T5TestCase(unittest.TestCase):
       def setUp(self) -> None:

              self.model = T5Model.init_from_json("./checkpoints/T5-base-ch/config.json")
              self.tokenizer = T5PegasusTokenizer("./checkpoints/T5-base-ch/vocab.txt")

              print("loading model successfully!")

       def test_model_predict(self):
              text = "一辆小轿车一名女司机竟造成9死24伤日前深圳市交警局对事故进行通报：" \
                     "从目前证据看事故系司机超速行驶且操作不当导致目前24名伤员已有6名治愈出院其余正接受治疗预计事故赔偿费或超一千万元"
              predictor = Predictor(self.model, self.tokenizer)
              out = predictor.predict_generate_randomsample(text, top_k=30, top_p=0.8, repetition_penalty=1.5, temperature=1.0, out_max_length=10)
              out_2 = predictor.predict_generate_beamsearch(text, beam_size=2, out_max_length=10)
              print(out)
              print(out_2)


def suite():
       suite = unittest.TestSuite()
       suite.addTest(T5TestCase('test_model_predict'))
       return suite


if __name__ == '__main__':
       runner = unittest.TextTestRunner()
       runner.run(suite())