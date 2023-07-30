# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import unittest
from flagai.data.tokenizer import Tokenizer
from flagai.auto_model.auto_loader import AutoLoader
import sys;sys.path.append("/home/yanzhaodong/FlagAI")

class TokenizerTestCase(unittest.TestCase):

    # def test_tokenizer_GLM_large_ch(self):
    #     tokenizer = Tokenizer.from_pretrained("GLM-large-ch")
    #     import pdb;pdb.set_trace()
    #     self.assertEqual(tokenizer.TokenToId("人"), 43371, 'Token id "人" error')
    #     self.assertEqual(tokenizer.EncodeAsIds("今天吃饭吃了肯德基"),
    #                      [3378, 1567, 2613, 20282], 'EncodeAsIds Error')
    #     self.assertEqual(tokenizer.DecodeIds([3378, 1567, 2613, 20282]),
    #                      '今天吃饭吃了肯德基', 'DecodeIds Error')
    #     self.assertEqual(tokenizer.tokenize('今天吃饭吃了肯德基'),
    #                      ['▁今天', '吃饭', '吃了', '肯德基'], 'tokenize Error')
    #     self.assertEqual(tokenizer.encode_plus('今天吃饭吃了肯德基')['input_ids'],
    #                      [50006, 3378, 1567, 2613, 20282, 50001], 'encode_plus Error')
    #     self.assertEqual(set([(k, v.token, v.Id) for k,v in tokenizer.command_name_map.items()]),
    #            {('pad', '<|endoftext|>', 50000), ('eos', '<|endoftext|>', 50000), ('sep', '[SEP]', 50001), 
    #            ('cls', '[CLS]', 50002), ('mask', '[MASK]', 50003), ('unk', '[UNK]', 50004), ('sop', '<|startofpiece|>', 50006),
    #             ('eop', '<|endofpiece|>', 50007), ('gMASK', '[gMASK]', 50007), ('sMASK', '[sMASK]', 50008)}, 'SpecialTokens error')

    # def test_tokenizer_GLM_large_en(self):
    #     tokenizer = Tokenizer.from_pretrained("GLM-large-en")
    #     import pdb;pdb.set_trace()
    #     self.assertEqual(tokenizer.TokenToId("day"), 2154, '')
    #     self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
    #                      [13017, 7975, 3084, 2033, 3407], '')
    #     self.assertEqual(tokenizer.DecodeIds([13017, 7975, 3084, 2033, 3407]),
    #                      'fried chicken makes me happy', 'DecodeIds Error')
    #     self.assertEqual(set([(k, v.token, v.Id) for k,v in tokenizer.command_name_map.items()]),
    #             {('pad', '[PAD]', 0), ('cls', '[CLS]', 101), ('mask', '[MASK]', 103), ('unk', '[UNK]', 100), 
    #             ('sep', '[SEP]', 102), ('eos', '[PAD]', 0), ('sop', '<|startofpiece|>', 30522), ('eop', '<|endofpiece|>', 30523), 
    #             ('gMASK', '[gMASK]', 30524), ('sMASK', '[sMASK]', 30525)})

    # def test_tokenizer_glm_10b_en(self):
    #     tokenizer = Tokenizer.from_pretrained("GLM-10b-en")
    #     self.assertEqual(tokenizer.TokenToId("day"), 820, '')
    #     self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
    #                      [25520, 9015, 1838, 502, 3772], '')
    #     self.assertEqual(tokenizer.DecodeIds([25520, 9015, 1838, 502, 3772]),
    #                      'fried chicken makes me happy', 'DecodeIds Error')
    #     self.assertEqual([(k, v.token, v.Id) for k,v in tokenizer.command_name_map.items()],
    #             [('eos', '[PAD]', 0), ('cls', '[CLS]', 101), ('mask', '[MASK]', 103), ('unk', '[UNK]', 100), 
    #             ('sep', '[SEP]', 102), ('pad', '[PAD]', 0), ('sop', '<|startofpiece|>', 30522), ('eop', '<|endofpiece|>', 30523), 
    #             ('gMASK', '[gMASK]', 30524), ('sMASK', '[sMASK]', 30525)])

    
    def test_tokenizer_t5(self):
        tokenizer = Tokenizer.from_pretrained('T5-base-ch')
        import pdb;pdb.set_trace()
        self.assertEqual(tokenizer.TokenToId("人"), 297, '')
        self.assertEqual(tokenizer.EncodeAsIds("今天吃饭吃了肯德基"),
                         [306, 1231, 798, 5447, 798, 266, 4017, 1738, 1166], '')
        self.assertEqual(tokenizer.DecodeIds([306, 1231, 798, 5447, 798, 266, 4017, 1738, 1166]),
                         '今天吃饭吃了肯德基', 'DecodeIds Error')
        encode_plus_result = tokenizer.encode_plus("今天吃饭吃了肯德基")
        self.assertEqual(list(encode_plus_result.keys()),
                         ['input_ids', 'token_type_ids'], 'encode_plus Error')
        self.assertEqual(encode_plus_result['input_ids'],
                    [101, 306, 1231, 798, 5447, 798, 266, 4017, 1738, 1166, 102], 'encode_plus Error')
        self.assertEqual(set([(k, v.token, v.Id) for k,v in tokenizer.command_name_map.items()]),
                 {('pad', '[PAD]', 0), ('cls', '[CLS]', 101), ('mask', '[MASK]', 103), ('unk', '[UNK]', 100), 
                 ('sep', '[SEP]', 102), ('eos', '[PAD]', 0), ('sop', '<|startofpiece|>', 50000), ('eop', '<|endofpiece|>', 50001), 
                 ('gMASK', '[gMASK]', 50002), ('sMASK', '[sMASK]', 50003)}, 'SpecialTokens error') 

        
    def test_tokenizer_roberta(self):
        tokenizer = Tokenizer.from_pretrained('RoBERTa-base-ch')
        import pdb;pdb.set_trace()
        self.assertEqual(tokenizer.TokenToId("人"), 782, '')
        self.assertEqual(tokenizer.EncodeAsIds("今天吃饭吃了肯德基"),
                         [791, 1921, 1391, 7649, 1391, 749, 5507, 2548, 1825], '')
        self.assertEqual(tokenizer.DecodeIds([791, 1921, 1391, 7649, 1391, 749, 5507, 2548, 1825]),
                         '今天吃饭吃了肯德基', 'DecodeIds Error')
        self.assertEqual(tokenizer.tokenize('今天吃饭吃了肯德基'),
                         ['今', '天', '吃', '饭', '吃', '了', '肯', '德', '基'], 'tokenize Error')
        self.assertEqual(tokenizer.encode_plus('今天吃饭吃了肯德基')['input_ids'],
                         [101, 791, 1921, 1391, 7649, 1391, 749, 5507, 2548, 1825, 102], 'encode_plus Error')
        self.assertEqual(set([(k, v.token, v.Id) for k,v in tokenizer.command_name_map.items()]),
                 {('pad', '[PAD]', 0), ('cls', '[CLS]', 101), ('mask', '[MASK]', 103), ('unk', '[UNK]', 100), 
                 ('sep', '[SEP]', 102), ('eos', '[PAD]', 0), ('sop', '<|startofpiece|>', 21128), 
                 ('eop', '<|endofpiece|>', 21129), ('gMASK', '[gMASK]', 21130), ('sMASK', '[sMASK]', 21131)}, 'SpecialTokens error')                

    def test_tokenizer_bert(self):
        tokenizer = Tokenizer.from_pretrained('BERT-base-en')
        self.assertEqual(tokenizer.TokenToId("day"), 2154, '')
        self.assertEqual(tokenizer.EncodeAsIds("fried chicken makes me happy"),
                         [13017, 7975, 3084, 2033, 3407], '')
        self.assertEqual(tokenizer.DecodeIds([13017, 7975, 3084, 2033, 3407]),
                         'fried chicken makes me happy', 'DecodeIds Error')
        self.assertEqual(tokenizer.tokenize('fried chicken makes me happy'),
                         ['fried', 'chicken', 'makes', 'me', 'happy'], 'tokenize Error')
        self.assertEqual(tokenizer.encode_plus('fried chicken makes me happy')['input_ids'],
                         [101, 13017, 7975, 3084, 2033, 3407, 102], 'encode_plus Error')
        self.assertEqual(set([(k, v.token, v.Id) for k,v in tokenizer.command_name_map.items()]),
                {('eos', '[PAD]', 0), ('unk', '[UNK]', 100), ('cls', '[CLS]', 101), ('sep', '[SEP]', 102), 
                 ('mask', '[MASK]', 103), ('pad', '[PAD]', 0),('sop', '<|startofpiece|>', 30522), 
                 ('eop', '<|endofpiece|>', 30523), ('gMASK', '[gMASK]', 30524), ('sMASK', '[sMASK]', 30525)}, 'SpecialTokens error')

    # def test_tokenizer_cpm1(self):
    #     loader = AutoLoader(task_name="lm",
    #                         model_name="CPM-large-ch",
    #                         model_dir="./checkpoints/",
    #                         only_download_config=True)
        
    #     tokenizer = loader.get_tokenizer()
    #     self.assertEqual(tokenizer.TokenToId("人"), 62, '')
    #     self.assertEqual(tokenizer.encode("今天吃饭吃了肯德基"),
    #                      [837, 3079, 1777, 3079, 139, 3687, 513, 1463], '')
    #     self.assertEqual(tokenizer.DecodeIds([837, 3079, 1777, 3079, 139, 3687, 513, 1463]),
    #                      '今天吃饭吃了肯德基', 'DecodeIds Error')
    #     self.assertEqual(tokenizer.tokenize('今天吃饭吃了肯德基'),
    #                      [837, 3079, 1777, 3079, 139, 3687, 513, 1463], 'tokenize Error')
    #     self.assertEqual(tokenizer.encode_plus('今天吃饭吃了肯德基')['input_ids'],
    #                      [837, 3079, 1777, 3079, 139, 3687, 513, 1463], 'encode_plus Error')
    #     self.assertEqual(set([(k, v.token, v.Id) for k,v in tokenizer.command_name_map.items()]),
    #              {('unk', '<unk>', 0), ('cls', '<s>', 1), ('eos', '</s>', 2), ('sep', '<sep>', 4), 
    #               ('mask', '<mask>', 6), ('pad', '<pad>', 5),('eod', '<eod>', 7)}, 'SpecialTokens error') 

    def test_tokenizer_opt(self):
        tokenizer = Tokenizer.from_pretrained('opt-1.3b-en')
        import pdb;pdb.set_trace()
        self.assertEqual(tokenizer.encode("day"), [1208], '')
        self.assertEqual(tokenizer.decode([21209, 5884, 817, 162, 1372]),
                         'fried chicken makes me happy', 'DecodeIds Error')
        self.assertEqual(tokenizer.tokenize('fried chicken makes me happy'),
                         ['fried', 'Ġchicken', 'Ġmakes', 'Ġme', 'Ġhappy'], 'tokenize Error')
        self.assertEqual(tokenizer.encode_plus('fried chicken makes me happy')['input_ids'],
                         [2, 21209, 5884, 817, 162, 1372], 'encode_plus Error')
        self.assertEqual(set([(k, v.token, v.Id) for k,v in tokenizer.command_name_map.items()]),
                {('cls', '<s>', 0), ('pad', '<pad>', 1), ('bos', '</s>', 2), ('eos', '</s>', 2), ('unk', '<unk>', 3),
                ('mask', '<mask>', 50264)}, 'SpecialTokens error')

    # def test_tokenizer_clip(self):
    #     loader = AutoLoader(task_name="txt_img_matching",
    #                 model_name="clip-base-p32-224")
    #     tokenizer = loader.get_tokenizer()
    #     self.assertEqual(tokenizer.tokenize_as_tensor("cat")[0][:3].tolist(), [49406, 2368, 49407], '')

    # def test_tokenizer_evaclip(self):
    #     loader = AutoLoader(task_name="txt_img_matching",
    #                 model_name="eva-clip")
    #     tokenizer = loader.get_tokenizer()
    #     self.assertEqual(tokenizer.tokenize_as_tensor("cat")[0][:3].tolist(), [49406, 2368, 49407], '')



def suite():
    suite = unittest.TestSuite()
    # suite.addTest(TokenizerTestCase('test_tokenizer_GLM_large_ch'))
    # suite.addTest(TokenizerTestCase('test_tokenizer_GLM_large_en'))
    # suite.addTest(TokenizerTestCase('test_tokenizer_glm_10_en'))
    suite.addTest(TokenizerTestCase('test_tokenizer_t5'))
    suite.addTest(TokenizerTestCase('test_tokenizer_roberta'))
    # suite.addTest(TokenizerTestCase('test_tokenizer_bert'))
    # suite.addTest(TokenizerTestCase('test_tokenizer_cpm1'))
    suite.addTest(TokenizerTestCase('test_tokenizer_opt'))
    # suite.addTest(TokenizerTestCase('test_tokenizer_clip'))
    # suite.addTest(TokenizerTestCase('test_tokenizer_evaclip'))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())