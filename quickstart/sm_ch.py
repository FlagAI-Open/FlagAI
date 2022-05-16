import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxlen = 256

auto_loader = AutoLoader("classification",
                         model_name="RoBERTa-wwm-ext-semantic-matching",   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
                         class_num=2)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()

predictor = Predictor(model, tokenizer)

test_data = [["后悔了吗", "你有没有后悔"], ["打开自动横屏", "开启移动数据"],
             ["我觉得你很聪明", "你聪明我是这么觉得"]]

for text_pair in test_data:
    print(text_pair, "相似" if predictor.predict_cls_classifier(text_pair) == 1 else "不相似")
