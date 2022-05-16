from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
import torch
import unittest


class AutoLoaderTestCase(unittest.TestCase):

    def setUp(self) -> None:

        self.task_name = [
            "seq2seq", "sequence_labeling", "classification",
            "sequence_labeling_crf", "sequence_labeling_gp", "embedding"
        ]
        self.model_name = [
            "GPT2_base_ch", "T5_base_ch", "RoBERTa-wwm-ext",
            "GLM_large_ch"
        ]

        print("loading bert model successfully!")

    def test_model_predict(self):
        for m_name in self.model_name:
            for t_name in self.task_name:
                if "gpt2" in m_name or "t5" in m_name:
                    if t_name != "seq2seq":
                        continue
                if "glm" in m_name and t_name not in ['seq2seq']:
                    continue
                loader = AutoLoader(task_name=t_name,
                                    model_name=m_name,
                                    class_num=3,
                                    inner_dim=32,
                                    only_download_config=True)
                model = loader.get_model()
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                tokenizer = loader.get_tokenizer()
                predictor = Predictor(model, tokenizer)
                text = "今天吃饭吃了肯德基"
                if t_name == "seq2seq":
                    output = predictor.predict_generate_beamsearch(
                        text, out_max_length=20)
                    output = predictor.predict_generate_randomsample(
                        text, out_max_length=20)
                elif t_name == "classification":
                    output = predictor.predict_cls_classifier(text)
                elif t_name == "sequence_labeling" or t_name == "sequence_labeling_crf" or t_name == "sequence_labeling_gp":
                    output = predictor.predict_ner(text,
                                                   target=["0", "1", "2"])
                elif t_name == "embedding":
                    output = predictor.predict_embedding(text)
                else:
                    output = None
                #assert output != None, "output is not None"
                print(
                    f"task_name is {t_name}, model_name is {m_name}, output is {output}"
                )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoLoaderTestCase('test_model_predict'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
