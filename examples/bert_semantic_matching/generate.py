import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_save_path = "./checkpoints_semantic_matching/11250/mp_rank_00_model_states.pt"

maxlen = 256

auto_loader = AutoLoader("classification",
                         model_name="RoBERTa-wwm-ext",
                         only_download_config=True,
                         classification_size=2)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()

predictor = Predictor(model, tokenizer)
model.load_state_dict(
    torch.load(model_save_path, map_location=device)["module"])

model.to(device)
model.eval()

test_data = [["后悔了吗", "你有没有后悔"], ["打开自动横屏", "开启移动数据"],
             ["我觉得你很聪明", "你聪明我是这么觉得"]]

for text_pair in test_data:
    print(predictor.predict_cls_classifier(text_pair))
