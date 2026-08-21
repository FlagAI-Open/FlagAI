# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.model.predictor.predictor import Predictor
import torch
from flagai.model.glm_model import GLMForSeq2Seq
from flagai.data.tokenizer import Tokenizer
import unittest

class GLMLargeChTestCase(unittest.TestCase):

    def setUp(self) -> None:

        self.model = GLMForSeq2Seq.init_from_json("./checkpoints/GLM-large-ch/config.json")
        self.tokenizer = Tokenizer.from_pretrained("GLM-large-ch")
        print("loading bert model successfully!")

    def test_model_predict(self):
        model = self.model
        tokenizer = self.tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        predictor = Predictor(model, tokenizer)
        text = "今天吃饭吃了肯德基"

        output_beam_search = predictor.predict_generate_beamsearch(text, out_max_length=20)
        output_randomsample = predictor.predict_generate_randomsample(text, out_max_length=20)

        print(f"output_beamsearch is {output_beam_search}")
        print(f"output_randomsample is {output_randomsample}")

def suite():
    suite = unittest.TestSuite()
    suite.addTest(GLMLargeChTestCase('test_model_predict'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
