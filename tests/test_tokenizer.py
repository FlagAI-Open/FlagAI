# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import unittest
from flagai.data.tokenizer import Tokenizer
from flagai.auto_model.auto_loader import AutoLoader

class TokenizerTestCase(unittest.TestCase):

    def test_tokenizer_GLM_large_ch(self):
        tokenizer = Tokenizer.from_pretrained("GLM-large-ch")
        self.assertEqual(tokenizer.TokenToId("人"), 43371, 'Token id "人" error')
        self.assertEqual(tokenizer.EncodeAsIds("今天吃饭吃了肯德基"),
                         [3378, 1567, 2613, 20282], 'EncodeAsIds Error')
        self.assertEqual(tokenizer.DecodeIds([3378, 1567, 2613, 20282]),
                         '今天吃饭吃了肯德基', 'DecodeIds Error')

    def test_tokenizer_GLM_large_en(self):
        tokenizer = Tokenizer.from_pretrained("GLM-large-en")
        self.assertEqual(tokenizer.TokenToId("day"), 2154, '')
        self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
                         [13017, 7975, 3084, 2033, 3407], '')
        self.assertEqual(tokenizer.DecodeIds([13017, 7975, 3084, 2033, 3407]),
                         'fried chicken makes me happy', 'DecodeIds Error')

    # def test_tokenizer_glm_10b_en(self):
    #     tokenizer = Tokenizer.from_pretrained("GLM-10b-en")
    #     self.assertEqual(tokenizer.TokenToId("day"), 820, '')
    #     self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
    #                      [25520, 9015, 1838, 502, 3772], '')
    #     self.assertEqual(tokenizer.DecodeIds([25520, 9015, 1838, 502, 3772]),
    #                      'fried chicken makes me happy', 'DecodeIds Error')
    
    def test_tokenizer_t5(self):
        tokenizer = Tokenizer.from_pretrained('t5-base-en')
        self.assertEqual(tokenizer.TokenToId("day"), 1135, '')
        self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
                         [3, 7704, 3832, 656, 140, 1095], '')
        self.assertEqual(tokenizer.DecodeIds([3, 7704, 3832, 656, 140, 1095]),
                         'fried chicken makes me happy', 'DecodeIds Error')

    def test_tokenizer_roberta(self):
        tokenizer = Tokenizer.from_pretrained('RoBERTa-base-ch')
        # print(tokenizer.DecodeIds([791, 1921, 1391, 7649, 1391, 749, 5507, 2548, 1825]))
        self.assertEqual(tokenizer.TokenToId("人"), 782, '')
        self.assertEqual(tokenizer.EncodeAsIds("今天吃饭吃了肯德基"),
                         [791, 1921, 1391, 7649, 1391, 749, 5507, 2548, 1825], '')
        self.assertEqual(tokenizer.DecodeIds([791, 1921, 1391, 7649, 1391, 749, 5507, 2548, 1825]),
                         '今天吃饭吃了肯德基', 'DecodeIds Error')

    def test_tokenizer_bert(self):
        tokenizer = Tokenizer.from_pretrained('BERT-base-en')
        self.assertEqual(tokenizer.TokenToId("day"), 2154, '')
        self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
                         [13017, 7975, 3084, 2033, 3407], '')
        self.assertEqual(tokenizer.DecodeIds([13017, 7975, 3084, 2033, 3407]),
                         'fried chicken makes me happy', 'DecodeIds Error')

    def test_tokenizer_cpm1(self):
        loader = AutoLoader(task_name="lm",
                            model_name="CPM-large-ch",
                            model_dir="./checkpoints/",
                            only_download_config=True)
        tokenizer = loader.get_tokenizer()
        self.assertEqual(tokenizer.encode("day"), [8, 8275], '')
        self.assertEqual(tokenizer.encode("fried chicken makes me happy"),
                         [2487, 27385, 9291, 9412, 3531, 14588, 289, 4406, 25239], '')
        self.assertEqual(tokenizer.decode([2487, 27385, 9291, 9412, 3531, 14588, 289, 4406, 25239]),
                         'fried chicken makes me happy', 'DecodeIds Error')

    def test_tokenizer_opt(self):
        tokenizer = Tokenizer.from_pretrained('opt-125m-en')
        self.assertEqual(tokenizer.encode("day"), [1208], '')
        self.assertEqual(tokenizer.encode_plus("fried chicken makes me happy")["input_ids"],
                         [50260, 21209, 5884, 817, 162, 1372, 50260], '')
        self.assertEqual(tokenizer.decode([21209, 5884, 817, 162, 1372]),
                         'fried chicken makes me happy', 'DecodeIds Error')

    def test_tokenizer_clip(self):
        loader = AutoLoader(task_name="txt_img_matching",
                    model_name="clip-base-p32-224")
        tokenizer = loader.get_tokenizer()
        self.assertEqual(tokenizer.tokenize_as_tensor("cat")[0][:3].tolist(), [49406, 2368, 49407], '')

    def test_tokenizer_evaclip(self):
        loader = AutoLoader(task_name="txt_img_matching",
                    model_name="eva-clip")
        tokenizer = loader.get_tokenizer()
        self.assertEqual(tokenizer.tokenize_as_tensor("cat")[0][:3].tolist(), [49406, 2368, 49407], '')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TokenizerTestCase('test_tokenizer_GLM_large_ch'))
    suite.addTest(TokenizerTestCase('test_tokenizer_GLM_large_en'))
    # suite.addTest(TokenizerTestCase('test_tokenizer_glm_10_en'))
    suite.addTest(TokenizerTestCase('test_tokenizer_t5'))
    suite.addTest(TokenizerTestCase('test_tokenizer_roberta'))
    suite.addTest(TokenizerTestCase('test_tokenizer_bert'))
    suite.addTest(TokenizerTestCase('test_tokenizer_cpm1'))
    suite.addTest(TokenizerTestCase('test_tokenizer_opt'))
    suite.addTest(TokenizerTestCase('test_tokenizer_clip'))
    suite.addTest(TokenizerTestCase('test_tokenizer_evaclip'))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())