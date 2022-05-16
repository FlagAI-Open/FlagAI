import unittest
from flagai.data.tokenizer import GLMLargeChTokenizer
from flagai.data.tokenizer import GLMLargeEnWordPieceTokenizer
from flagai.data.tokenizer import GLM10bENBPETokenizer
from flagai.data.tokenizer import T5BPETokenizer
from flagai.data.tokenizer import ROBERTATokenizer
from flagai.data.tokenizer import BertWordPieceTokenizer


class TokenizerTestCase(unittest.TestCase):

    def test_tokenizer_glm_chinesesp(self):
        tokenizer = GLMLargeChTokenizer(fix_command_token=True,
                                        add_block_symbols=True,
                                        add_task_mask=False,
                                        add_decoder_mask=False)

        self.assertEqual(tokenizer.TokenToId("人"), 43371, 'Token id "人" error')
        self.assertEqual(tokenizer.EncodeAsIds("今天吃饭吃了肯德基"),
                         [3378, 1567, 2613, 20282], 'EncodeAsIds Error')
        self.assertEqual(tokenizer.DecodeIds([3378, 1567, 2613, 20282]),
                         '今天吃饭吃了肯德基', 'DecodeIds Error')

    def test_tokenizer_glm_bertwordpiece(self):
        tokenizer = GLMLargeEnWordPieceTokenizer()
        print(tokenizer.EncodeAsIds("today is a nice day and"))
        self.assertEqual(tokenizer.TokenToId("day"), 2154, '')
        self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
                         [13017, 7975, 3084, 2033, 3407], '')
        self.assertEqual(tokenizer.DecodeIds([13017, 7975, 3084, 2033, 3407]),
                         'fried chicken makes me happy', 'DecodeIds Error')

    def test_tokenizer_glm_gpt2(self):
        tokenizer = GLM10bENBPETokenizer()
        self.assertEqual(tokenizer.TokenToId("day"), 820, '')
        self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
                         [25520, 9015, 1838, 502, 3772], '')
        self.assertEqual(tokenizer.DecodeIds([25520, 9015, 1838, 502, 3772]),
                         'fried chicken makes me happy', 'DecodeIds Error')

    def test_tokenizer_t5(self):
        tokenizer = T5BPETokenizer(tokenizer_model_type='t5-base')
        self.assertEqual(tokenizer.TokenToId("day"), 1135, '')
        self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
                         [3, 7704, 3832, 656, 140, 1095], '')
        self.assertEqual(tokenizer.DecodeIds([3, 7704, 3832, 656, 140, 1095]),
                         'fried chicken makes me happy', 'DecodeIds Error')

    def test_tokenizer_roberta(self):
        tokenizer = ROBERTATokenizer(tokenizer_model_type='roberta-base')
        self.assertEqual(tokenizer.TokenToId("day"), 1208, '')
        self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
                         [21209, 5884, 817, 162, 1372], '')
        self.assertEqual(tokenizer.DecodeIds([21209, 5884, 817, 162, 1372]),
                         'fried chicken makes me happy', 'DecodeIds Error')

    def test_tokenizer_bert(self):
        tokenizer = BertWordPieceTokenizer(
            tokenizer_model_type='bert-large-uncased')
        self.assertEqual(tokenizer.TokenToId("day"), 2154, '')
        self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
                         [13017, 7975, 3084, 2033, 3407], '')
        self.assertEqual(tokenizer.DecodeIds([13017, 7975, 3084, 2033, 3407]),
                         'fried chicken makes me happy', 'DecodeIds Error')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TokenizerTestCase('test_tokenizer_glm_chinesesp'))
    suite.addTest(TokenizerTestCase('test_tokenizer_glm_bertwordpiece'))
    suite.addTest(TokenizerTestCase('test_tokenizer_glm_gpt2'))
    suite.addTest(TokenizerTestCase('test_tokenizer_t5'))
    suite.addTest(TokenizerTestCase('test_tokenizer_roberta'))
    suite.addTest(TokenizerTestCase('test_tokenizer_bert'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
